let rpSchema = (window.APP_BOOTSTRAP && window.APP_BOOTSTRAP.rpSchema) || {};
const isFileMode = window.location.protocol === "file:";
const respondentKey = "survey_respondent_id";
const deviceKey = "survey_device_tag";

let currentAssignment = null;
let respondentCandidates = [];
let designMeta = {
  save_name: "",
  sp_intro_text: "",
  conditions: [],
  byAltVar: {},
  byVar: {},
};

function genRespondentId() {
  const getOrCreateDeviceTag = () => {
    const old = String(localStorage.getItem(deviceKey) || "").trim();
    if (old) return old;
    const ua = (navigator.userAgent || "").toLowerCase();
    const platform = (navigator.platform || "").toLowerCase();
    const os = platform.includes("mac") ? "MAC" : (platform.includes("win") ? "WIN" : (platform.includes("linux") ? "LNX" : "UNK"));
    const browser = ua.includes("edg") ? "EDG" : (ua.includes("chrome") ? "CHR" : (ua.includes("safari") ? "SAF" : (ua.includes("firefox") ? "FF" : "BRW")));
    const form = /iphone|ipad|android|mobile/.test(ua) ? "MOB" : "PC";
    const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
    const tag = `${os}${browser}${form}_${rand}`;
    localStorage.setItem(deviceKey, tag);
    return tag;
  };
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  const r = Math.random().toString(36).slice(2, 6).toUpperCase();
  return `R${ts}_${getOrCreateDeviceTag()}_${r}`;
}

function getOrCreateRespondentId() {
  const old = String(localStorage.getItem(respondentKey) || "").trim();
  if (old) return old;
  const id = genRespondentId();
  localStorage.setItem(respondentKey, id);
  return id;
}

function clearAllSurveyCache() {
  [
    "survey_profile_data",
    "survey_trip_diary",
    "survey_map_picker_result",
    "survey_map_picker_context",
    "survey_sp_current_assignment",
  ].forEach((k) => localStorage.removeItem(k));
}

function showActionDialog(title, message, actions) {
  return new Promise((resolve) => {
    const mask = document.createElement("div");
    mask.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:9999;display:flex;align-items:center;justify-content:center;padding:16px;";
    const box = document.createElement("div");
    box.style.cssText = "background:#fff;max-width:560px;width:100%;border-radius:10px;padding:14px 14px 12px;border:1px solid #d6e1ee;";
    const h = document.createElement("h3");
    h.textContent = title;
    h.style.margin = "0 0 8px";
    const p = document.createElement("p");
    p.textContent = message;
    p.style.margin = "0 0 12px";
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
    actions.forEach((a) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = a.label;
      b.style.cssText = "min-height:38px;padding:8px 12px;";
      b.addEventListener("click", () => {
        mask.remove();
        resolve(a.id);
      });
      row.appendChild(b);
    });
    box.appendChild(h);
    box.appendChild(p);
    box.appendChild(row);
    mask.appendChild(box);
    document.body.appendChild(mask);
  });
}

function updateSpInheritBanner() {
  const banner = document.getElementById("spInheritBanner");
  if (!banner) return;
  const rid = getOrCreateRespondentId();
  const hasProfile = !!localStorage.getItem("survey_profile_data");
  const hasTrip = !!localStorage.getItem("survey_trip_diary");
  let src = "无";
  if (hasProfile && hasTrip) src = "Profile + TripDiary";
  else if (hasProfile) src = "Profile";
  else if (hasTrip) src = "TripDiary";
  const saveName = designMeta && designMeta.save_name ? `；design=${designMeta.save_name}` : "";
  banner.textContent = `继承检查：来源=${src}；respondent_id=${rid}${saveName}`;
}

function exportCurrentRespondentJson() {
  const payload = {
    respondent_id: getOrCreateRespondentId(),
    exported_at: new Date().toISOString(),
    profile: (() => { try { return JSON.parse(localStorage.getItem("survey_profile_data") || "null"); } catch (_e) { return null; } })(),
    trip_diary: (() => { try { return JSON.parse(localStorage.getItem("survey_trip_diary") || "null"); } catch (_e) { return null; } })(),
    sp: {
      current_assignment: (() => { try { return JSON.parse(localStorage.getItem("survey_sp_current_assignment") || "null"); } catch (_e) { return null; } })(),
      last_submission: (() => { try { return JSON.parse(localStorage.getItem("survey_sp_last_submission") || "null"); } catch (_e) { return null; } })(),
      current_rp: collectRp(),
      selected_design_save_name: designMeta.save_name || null,
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${payload.respondent_id || "respondent"}_survey_export.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function normalizeKey(s) {
  return String(s || "").trim().toLowerCase().replace(/\s+/g, "_");
}

function resetDesignMeta() {
  const fallbackLabels = {
    time: "出行时间（分钟）",
    cost: "出行费用（元）",
    wait_time: "等待时间（分钟）",
    walk_distance: "步行距离（米）",
    park_time: "停车位寻找时间（分钟）",
  };
  designMeta = {
    save_name: "",
    sp_intro_text: "",
    conditions: [],
    byAltVar: {},
    byVar: { ...fallbackLabels },
  };
}

function applyDesignPayload(payload) {
  const spec = (payload && payload.design_spec) || {};

  const intro = String(spec.sp_intro_text || "").trim();
  if (intro) designMeta.sp_intro_text = intro;

  const conditions = Array.isArray(spec.conditions)
    ? spec.conditions.map((x) => String(x || "").trim()).filter((x) => x.length > 0)
    : [];
  if (conditions.length) designMeta.conditions = conditions;

  (spec.alternatives || []).forEach((alt) => {
    const altName = normalizeKey(alt.name);
    (alt.variables || []).forEach((v) => {
      const varName = normalizeKey(v.name);
      const desc = String(v.description || "").trim();
      if (!varName) return;
      if (desc) {
        designMeta.byVar[varName] = desc;
        designMeta.byAltVar[`${altName}.${varName}`] = desc;
      }
    });
  });
}

function variableLabel(mode, key) {
  const altKey = normalizeKey(mode);
  const varKey = normalizeKey(key);
  return designMeta.byAltVar[`${altKey}.${varKey}`] || designMeta.byVar[varKey] || key;
}

function normalizeModeName(raw) {
  const n = normalizeKey(raw);
  if (n === "pt" || n === "publictransit" || n === "transit") return "public_transit";
  return n;
}

function resolveConditionOperand(task, token) {
  const t = String(token || "").trim();
  if (!t) return { ok: false, value: 0 };
  if (/^-?\d+(\.\d+)?$/.test(t)) return { ok: true, value: Number(t) };

  const dot = t.indexOf(".");
  if (dot <= 0 || dot >= t.length - 1) return { ok: false, value: 0 };
  const alt = normalizeModeName(t.slice(0, dot));
  const attr = normalizeKey(t.slice(dot + 1));
  const attrs = task.alternatives && task.alternatives[alt];
  if (!attrs) return { ok: false, value: 0 };
  if (attrs[attr] === undefined || attrs[attr] === null || attrs[attr] === "") return { ok: false, value: 0 };

  const numeric = Number(attrs[attr]);
  if (Number.isNaN(numeric)) return { ok: false, value: 0 };
  return { ok: true, value: numeric };
}

function compareValues(left, op, right) {
  if (op === ">") return left > right;
  if (op === ">=") return left >= right;
  if (op === "<") return left < right;
  if (op === "<=") return left <= right;
  return true;
}

function taskSatisfiesConditions(task) {
  const conditions = designMeta.conditions || [];
  if (!conditions.length) return true;

  for (const line of conditions) {
    const parts = line.split(/(>=|<=|>|<)/).map((x) => x.trim()).filter((x) => x.length > 0);
    if (parts.length < 3 || parts.length % 2 === 0) continue;

    let ok = true;
    for (let i = 1; i < parts.length; i += 2) {
      const left = resolveConditionOperand(task, parts[i - 1]);
      const op = parts[i];
      const right = resolveConditionOperand(task, parts[i + 1]);
      if (!left.ok || !right.ok || !compareValues(left.value, op, right.value)) {
        ok = false;
        break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

function createRpFields() {
  const root = document.getElementById("rpFields");
  if (!root) return;
  root.innerHTML = "";

  Object.entries(rpSchema).forEach(([key, values]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.textContent = key;

    if (Array.isArray(values)) {
      const select = document.createElement("select");
      select.id = `rp_${key}`;
      values.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = String(v);
        opt.textContent = String(v);
        select.appendChild(opt);
      });
      wrapper.appendChild(label);
      wrapper.appendChild(select);
    }

    root.appendChild(wrapper);
  });
}
window.createRpFields = createRpFields;

function setSchema(newSchema) {
  if (newSchema && typeof newSchema === "object") rpSchema = newSchema;
  createRpFields();
  updateSpInheritBanner();
}
window.setSurveySchema = setSchema;

function collectRp() {
  const rp = {};
  Object.keys(rpSchema).forEach((key) => {
    const el = document.getElementById(`rp_${key}`);
    if (!el) return;
    const value = el.value;
    if (key === "family_size" || key === "car_availability") {
      rp[key] = Number(value);
    } else {
      rp[key] = value;
    }
  });
  return rp;
}

function optionValuesFor(key) {
  const el = document.getElementById(`rp_${key}`);
  if (!el) return [];
  return Array.from(el.options).map((o) => String(o.value));
}

function pickOption(key, preferred, fallbacks = []) {
  const values = optionValuesFor(key);
  const all = [preferred, ...fallbacks].map((x) => String(x == null ? "" : x));
  for (const v of all) {
    if (values.includes(v)) return v;
  }
  return values.length ? values[0] : "";
}

function normalizeIncomeToBucket(raw) {
  const s = String(raw || "").trim();
  if (!s) return "";
  if (s.includes("暂无") || s.includes("0~6万")) return "<5k";
  if (s.includes("6~12万")) return "5k-10k";
  if (s.includes("12~24万")) return "10k-20k";
  if (s.includes("24") || s.includes("36") || s.includes("48") || s.includes("72")) return ">20k";
  return "";
}

function normalizeAgeToBucket(raw) {
  const s = String(raw || "").trim();
  if (!s) return "";
  if (s.includes("18~30")) return "18-30";
  if (s.includes("30~45")) return "31-45";
  if (s.includes("45~60")) return "46-60";
  if (s.includes("60")) return "60+";
  return "";
}

function normalizeOccupation(raw) {
  const s = String(raw || "").trim();
  if (!s) return "";
  if (s.includes("学生")) return "student";
  if (s.includes("服务")) return "service";
  if (s.includes("机关") || s.includes("行政") || s.includes("技术") || s.includes("教师") || s.includes("医生") || s.includes("工程师")) return "office";
  return "other";
}

function deriveRpFromMember(member, profile) {
  const familySize = Number(profile && profile.family_total ? profile.family_total : ((profile && profile.household_members && profile.household_members.length) || 1));
  const carUsage = String((member && member.car_usage) || "");
  const carAvailability = carUsage && !carUsage.includes("无小汽车") ? 1 : 0;
  const incomeRaw = (member && member.income_pre_tax) || (profile && profile.household_income_annual) || "";
  return {
    age_group: pickOption("age_group", normalizeAgeToBucket(member && member.age_group), ["31-45", "18-30"]),
    income_group: pickOption("income_group", normalizeIncomeToBucket(incomeRaw), ["5k-10k"]),
    gender: pickOption("gender", member && member.gender === "女" ? "female" : "male", ["male"]),
    family_size: familySize,
    car_availability: carAvailability,
    occupation: pickOption("occupation", normalizeOccupation(member && member.occupation), ["other"]),
  };
}

function buildRespondentCandidates() {
  const profileRaw = localStorage.getItem("survey_profile_data");
  const tripRaw = localStorage.getItem("survey_trip_diary");
  let profile = {};
  let trip = {};
  try { profile = profileRaw ? JSON.parse(profileRaw) : {}; } catch (_e) {}
  try { trip = tripRaw ? JSON.parse(tripRaw) : {}; } catch (_e) {}

  const profileMembers = Array.isArray(profile.household_members) ? profile.household_members : [];
  const tripMembers = Array.isArray(trip.members) ? trip.members : [];
  const sourceMembers = tripMembers.length ? tripMembers : profileMembers;

  respondentCandidates = sourceMembers.map((m, idx) => {
    const memberId = m && m.member_id ? String(m.member_id) : `P${idx + 1}`;
    const role = m && m.relationship ? String(m.relationship) : "";
    const age = m && m.age_group ? String(m.age_group) : "";
    const label = `${memberId}${role ? ` / ${role}` : ""}${age ? ` / ${age}` : ""}`;
    return { member_id: memberId, label, rp: deriveRpFromMember(m || {}, profile || {}) };
  });
}

function hasCurrentProfileData() {
  const raw = localStorage.getItem("survey_profile_data");
  if (!raw) return false;
  try {
    const obj = JSON.parse(raw);
    return !!(obj && typeof obj === "object" && Object.keys(obj).length > 0);
  } catch (_e) {
    return false;
  }
}

function applyRespondentRpByMember(memberId) {
  let target = null;
  if (memberId) target = respondentCandidates.find((x) => x.member_id === memberId) || null;
  if (!target) target = respondentCandidates.find((x) => x.member_id === "P1") || respondentCandidates[0] || null;
  if (!target) return;
  const rp = target.rp || {};
  Object.entries(rp).forEach(([k, v]) => {
    const el = document.getElementById(`rp_${k}`);
    if (!el) return;
    el.value = String(v);
  });
}

function initRespondentInheritance() {
  buildRespondentCandidates();
  const select = document.getElementById("respondentSourceSelect");
  const inheritBtn = document.getElementById("inheritRespondentBtn");
  if (!select || !inheritBtn) return;

  select.innerHTML = '<option value="">自动选择(P1优先)</option>';
  respondentCandidates.forEach((c) => {
    const op = document.createElement("option");
    op.value = c.member_id;
    op.textContent = c.label;
    select.appendChild(op);
  });

  inheritBtn.addEventListener("click", () => applyRespondentRpByMember(select.value));
  select.addEventListener("change", () => applyRespondentRpByMember(select.value));
  applyRespondentRpByMember(select.value);
}

function renderTasks(tasks) {
  const box = document.getElementById("tasksContainer");
  if (!box) return;
  box.innerHTML = "";
  const filteredTasks = tasks.filter((task) => taskSatisfiesConditions(task));

  if (currentAssignment && currentAssignment.preview_only) {
    const preview = document.createElement("div");
    preview.className = "task preview-banner";
    const h = document.createElement("h3");
    h.textContent = "设计预览模式";
    const p = document.createElement("p");
    p.textContent = "当前题组只用于查看 SP 问卷样式，不绑定 respondent_id，不写入服务器分发记录，也不能提交为真实答案。";
    preview.appendChild(h);
    preview.appendChild(p);
    box.appendChild(preview);
  }

  if (designMeta.sp_intro_text) {
    const intro = document.createElement("div");
    intro.className = "task";
    const p = document.createElement("p");
    p.textContent = designMeta.sp_intro_text;
    intro.appendChild(p);
    box.appendChild(intro);
  }

  if (designMeta.conditions.length) {
    const cond = document.createElement("div");
    cond.className = "task";
    const p = document.createElement("p");
    p.textContent = `已启用条件约束：${designMeta.conditions.join(" ; ")}`;
    cond.appendChild(p);
    box.appendChild(cond);
  }

  if (filteredTasks.length < tasks.length) {
    const note = document.createElement("div");
    note.className = "task";
    const p = document.createElement("p");
    p.textContent = `已按条件筛除 ${tasks.length - filteredTasks.length} 个不合理组合。`;
    note.appendChild(p);
    box.appendChild(note);
  }

  filteredTasks.forEach((task, idx) => {
    const taskDiv = document.createElement("div");
    taskDiv.className = "task";
    taskDiv.dataset.taskId = task.id;

    const title = document.createElement("h3");
    title.textContent = `题目 ${idx + 1} (${task.id})`;
    taskDiv.appendChild(title);

    const altGrid = document.createElement("div");
    altGrid.className = "alt-grid";

    Object.entries(task.alternatives).forEach(([mode, attrs], altIdx) => {
      const alt = document.createElement("div");
      alt.className = "alt";
      alt.innerHTML = `<h4>${mode}</h4>`;

      const ul = document.createElement("ul");
      Object.entries(attrs).forEach(([k, v]) => {
        const li = document.createElement("li");
        li.textContent = `${variableLabel(mode, k)}: ${v}`;
        ul.appendChild(li);
      });
      alt.appendChild(ul);

      const choose = document.createElement("label");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `choice_${task.id}`;
      radio.value = mode;
      // Set required on one radio in each group so browser enforces one choice per SP task.
      if (altIdx === 0) radio.required = true;
      choose.appendChild(radio);
      choose.appendChild(document.createTextNode(" 选择该方式"));
      alt.appendChild(choose);

      altGrid.appendChild(alt);
    });

    taskDiv.appendChild(altGrid);
    box.appendChild(taskDiv);
  });
}

function setSubmitButtonMode(previewOnly) {
  const submitBtn = document.getElementById("submitBtn");
  if (!submitBtn) return;
  submitBtn.disabled = !!previewOnly;
  submitBtn.textContent = previewOnly ? "预览模式不提交" : "提交答案";
  submitBtn.title = previewOnly ? "设计预览不写入服务器；请保存 RP/Profile 后重新获取正式题组再提交。" : "";
}

function maybeToNumber(v) {
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}

function buildRealtimeTasksFromSavedSpec(payload, recommendation) {
  const spec = (payload && payload.design_spec) || {};
  const alts = Array.isArray(spec.alternatives) ? spec.alternatives : [];
  const designType = String((payload && payload.design_type) || "efficient");
  const opts = (payload && payload.design_options) || {};
  let tasksPerPerson = Number((recommendation && recommendation.tasks_per_person) || (payload.tasks_per_person_candidates || [6])[0] || 6);
  if (designType === "efficient" && opts.efficient && opts.efficient.tasks_per_person) {
    tasksPerPerson = Number(opts.efficient.tasks_per_person);
  }
  const rlOpts = designType === "selfattention" ? (opts.selfattention || {}) : (opts.dyppo || {});
  if ((designType === "dyppo" || designType === "selfattention") && rlOpts && rlOpts.tasks_per_round) {
    tasksPerPerson = Number(rlOpts.tasks_per_round);
  }
  const nTasks = Math.max(1, tasksPerPerson);
  const eps = Number((rlOpts && rlOpts.explore_epsilon) || 0.2);

  const tasks = [];
  for (let t = 0; t < nTasks; t += 1) {
    const alternatives = {};
    alts.forEach((alt, altIdx) => {
      const altName = normalizeKey(alt.name);
      if (!altName) return;
      const attrs = {};
      (alt.variables || []).forEach((v, varIdx) => {
        const varName = normalizeKey(v.name);
        const levels = Array.isArray(v.levels) ? v.levels : [];
        if (!varName || !levels.length) return;
        let lv = levels[(t + altIdx + varIdx) % levels.length];
        if ((designType === "dyppo" || designType === "selfattention") && Math.random() < eps) {
          const offset = Math.floor(Math.random() * levels.length);
          lv = levels[(t + altIdx + varIdx + offset) % levels.length];
        }
        attrs[varName] = maybeToNumber(lv);
      });
      alternatives[altName] = attrs;
    });
    tasks.push({ id: `saved_${Date.now()}_${t + 1}`, alternatives });
  }
  return tasks;
}

function setGeneratedAssignment(tasks, recommendation, createdAt, source, meta = {}) {
  const rid = String(meta.respondent_id || getOrCreateRespondentId());
  const assignmentId = String(meta.assignment_id || `saved_design_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
  const policyVersion = meta.policy_version == null ? "saved_design" : meta.policy_version;
  const previewOnly = !!meta.preview_only;
  currentAssignment = {
    assignment_id: assignmentId,
    respondent_id: rid,
    policy_version: policyVersion,
    preview_only: previewOnly,
    tasks: tasks.filter((task) => taskSatisfiesConditions(task)),
  };
  localStorage.setItem(
    "survey_sp_current_assignment",
    JSON.stringify({
      saved_at: new Date().toISOString(),
      source,
      respondent_id: rid,
      preview_only: previewOnly,
      payload: currentAssignment,
    }),
  );

  // Frontend rendering never writes distribution logs. The formal server issue route records
  // real assignments; the preview endpoint deliberately does not.

  renderTasks(tasks);
  const spSection = document.getElementById("spSection");
  if (spSection) {
    spSection.classList.remove("hidden");
    spSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  setSubmitButtonMode(previewOnly);

  const metrics = document.getElementById("metricsBox");
  if (metrics) {
    metrics.textContent = JSON.stringify(
      {
        mode: source,
        design_type: source.includes("selfattention") ? "selfattention" : (source.includes("dyppo") ? "dyppo" : "efficient"),
        save_name: designMeta.save_name || null,
        created_at: createdAt || null,
        assignment_id: assignmentId,
        preview_only: previewOnly,
        policy_version: policyVersion,
        recommendation: recommendation || null,
        tasks_total: tasks.length,
        tasks_after_condition: currentAssignment.tasks.length,
      },
      null,
      2,
    );
  }

  if (!currentAssignment.tasks.length) {
    alert("当前约束条件过严，题组被全部筛除。请调整条件后重试。");
  }
  updateSpInheritBanner();
}

async function loadDesignOptions() {
  const select = document.getElementById("spDesignSelect");
  if (!select) return;
  select.innerHTML = '<option value="">请先选择</option>';

  if (isFileMode) {
    const raw = localStorage.getItem("survey_sp_design_payload");
    if (raw) {
      const op = document.createElement("option");
      op.value = "local_saved_design";
      op.textContent = "local_saved_design";
      select.appendChild(op);
    }
    return;
  }

  try {
    const res = await fetch("/api/design/spec");
    const data = await res.json();
    if (!res.ok || !data) return;

    const names = Array.isArray(data.available_save_names)
      ? data.available_save_names
      : (data.save_name ? [data.save_name] : []);

    names.forEach((name) => {
      const n = String(name || "").trim();
      if (!n) return;
      const op = document.createElement("option");
      op.value = n;
      op.textContent = n;
      select.appendChild(op);
    });

    const latest = String(data.latest_save_name || data.save_name || "").trim();
    if (latest) select.value = latest;
  } catch (_e) {}
}

function collectChoices() {
  if (!currentAssignment) return {};
  const choices = {};
  currentAssignment.tasks.forEach((task) => {
    const selected = document.querySelector(`input[name="choice_${task.id}"]:checked`);
    if (selected) choices[task.id] = selected.value;
  });
  return choices;
}

async function fetchMetrics() {
  if (isFileMode) {
    const box = document.getElementById("metricsBox");
    if (box) {
      box.textContent = JSON.stringify(
        { mode: "file_preview", note: "静态直开模式不连接后端API" },
        null,
        2,
      );
    }
    return;
  }
  const res = await fetch("/api/design/metrics");
  const data = await res.json();
  document.getElementById("metricsBox").textContent = JSON.stringify(data, null, 2);
  document.getElementById("policyVersion").textContent = data.policy_version;
}

async function computeTasksFromServer(payload) {
  const create = await fetch("/api/design/compute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const createData = await create.json();
  if (!create.ok || createData.error || !createData.job_id) {
    throw new Error(createData.error || "提交计算任务失败");
  }
  const jobId = createData.job_id;
  const statusUrl = createData.status_url || `/api/design/compute/${encodeURIComponent(jobId)}`;
  let resultUrl = `/api/design/compute/${encodeURIComponent(jobId)}/result`;
  for (let i = 0; i < 240; i += 1) {
    await new Promise((r) => setTimeout(r, 500));
    const res = await fetch(statusUrl);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "读取计算进度失败");
    if (data.result_url) resultUrl = data.result_url;
    if (data.status === "done") {
      const rr = await fetch(resultUrl);
      const rd = await rr.json();
      if (!rr.ok || rd.error) throw new Error(rd.error || "拉取计算结果失败");
      return rd || {};
    }
    if (data.status === "failed") throw new Error(data.error || "计算失败");
  }
  throw new Error("计算超时，请重试");
}

async function issueTasksFromServer(designSaveName) {
  const rid = getOrCreateRespondentId();
  const res = await fetch("/api/design/issue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      design_save_name: designSaveName,
      respondent_id: rid,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    const err = new Error(data.error || "获取SP题组失败");
    err.status = res.status;
    err.data = data || {};
    throw err;
  }
  return data || {};
}

async function previewDesignFromServer(designSaveName) {
  const res = await fetch("/api/design/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ design_save_name: designSaveName }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    const err = new Error(data.error || "预览SP题组失败");
    err.status = res.status;
    err.data = data || {};
    throw err;
  }
  return data || {};
}

async function previewDesignOnly(selectedName, specData = null) {
  resetDesignMeta();
  if (isFileMode) {
    const raw = localStorage.getItem("survey_sp_design_payload");
    if (!raw) {
      alert("未找到本地保存的SP设计数据，请先到SP设计页计算并保存。");
      return;
    }
    const payload = JSON.parse(raw);
    designMeta.save_name = selectedName || "local_saved_design";
    applyDesignPayload(payload);
    const tasks = buildRealtimeTasksFromSavedSpec(payload, null);
    setGeneratedAssignment(tasks, null, new Date().toISOString(), "preview_local_saved_design", {
      assignment_id: `preview_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      respondent_id: "PREVIEW_ONLY",
      preview_only: true,
      policy_version: "preview",
    });
    return;
  }

  const data = specData || {};
  if (data.payload) {
    designMeta.save_name = String(data.save_name || selectedName);
    applyDesignPayload(data.payload);
    try {
      localStorage.setItem("survey_sp_design_payload", JSON.stringify(data.payload));
    } catch (_e) {}
  } else {
    designMeta.save_name = selectedName;
  }

  const preview = await previewDesignFromServer(designMeta.save_name || selectedName);
  const tasks = Array.isArray(preview.tasks) ? preview.tasks : [];
  const designType = String(preview.design_type || (data.payload && data.payload.design_type) || "efficient");
  setGeneratedAssignment(
    tasks,
    preview.recommendation || data.recommendation || null,
    preview.previewed_at || preview.issued_at || new Date().toISOString(),
    designType === "selfattention"
      ? "preview_server_saved_design_selfattention"
      : (designType === "dyppo" ? "preview_server_saved_design_dyppo" : "preview_server_saved_design_efficient"),
    {
      assignment_id: preview.assignment_id,
      respondent_id: "PREVIEW_ONLY",
      preview_only: true,
      policy_version: "preview",
    },
  );
  updateSpInheritBanner();
}

async function loadDesign() {
  const select = document.getElementById("spDesignSelect");
  const selectedName = String((select && select.value) || "").trim();
  if (!selectedName) {
    alert("请先选择SP设计版本（save_name）。");
    return;
  }

  resetDesignMeta();

  if (isFileMode) {
    const raw = localStorage.getItem("survey_sp_design_payload");
    if (!raw) {
      alert("未找到本地保存的SP设计数据，请先到SP设计页计算并保存。");
      return;
    }
    const payload = JSON.parse(raw);
    designMeta.save_name = selectedName;
    applyDesignPayload(payload);
    const tasks = buildRealtimeTasksFromSavedSpec(payload, null);
    setGeneratedAssignment(tasks, null, new Date().toISOString(), "generated_from_local_saved_design");
    return;
  }

  let specData = null;
  try {
    const res = await fetch(`/api/design/spec?save_name=${encodeURIComponent(selectedName)}`);
    const data = await res.json();
    if (!res.ok || !data || !data.payload) {
      alert("读取保存的SP设计失败，请确认保存名是否存在。");
      return;
    }
    specData = data;

    designMeta.save_name = String(data.save_name || selectedName);
    applyDesignPayload(data.payload);
    try {
      localStorage.setItem("survey_sp_design_payload", JSON.stringify(data.payload));
    } catch (_e) {}

    const designType = String(data.payload.design_type || "efficient");
    const issue = await issueTasksFromServer(designMeta.save_name || selectedName);
    const tasks = Array.isArray(issue.tasks) ? issue.tasks : [];
    const rec = issue.recommendation || data.recommendation || null;
    setGeneratedAssignment(
      tasks,
      rec,
      issue.issued_at || data.saved_at || new Date().toISOString(),
      designType === "selfattention"
        ? "issued_from_server_saved_design_selfattention"
        : (designType === "dyppo" ? "issued_from_server_saved_design_dyppo" : "issued_from_server_saved_design_efficient"),
      {
        assignment_id: issue.assignment_id,
        respondent_id: issue.respondent_id,
        policy_version: issue.policy_version,
      },
    );
    updateSpInheritBanner();
  } catch (_e) {
    const requiresProfile = _e && _e.status === 409 && _e.data && _e.data.requires_profile;
    if (requiresProfile) {
      const action = await showActionDialog(
        "需要先保存RP/Profile",
        "正式下发 SP 题组需要绑定当前 respondent_id，并记录到对应受访者 JSON。设计阶段如果只想查看题面样式，可以选择“仅预览问卷样式”。",
        [
          { id: "go_profile", label: "去保存RP/Profile" },
          { id: "preview", label: "仅预览问卷样式" },
          { id: "cancel", label: "取消" },
        ],
      );
      if (action === "go_profile") {
        window.location.href = isFileMode ? "profile.html" : "/survey/profile";
        return;
      }
      if (action === "preview") {
        try {
          await previewDesignOnly(selectedName, specData);
        } catch (previewErr) {
          alert(previewErr && previewErr.message ? previewErr.message : "预览SP题组失败。");
        }
      }
      return;
    }
    alert(_e && _e.message ? _e.message : "读取保存的SP设计失败，请稍后重试。");
  }
}

async function submitResponse() {
  if (isFileMode) {
    alert("当前是静态直开模式，无法提交到后端接口。");
    return;
  }
  if (!currentAssignment) {
    alert("请先获取题组。");
    return;
  }
  if (currentAssignment.preview_only || String(currentAssignment.assignment_id || "").startsWith("preview_")) {
    alert("当前是设计预览模式，不会写入服务器。请先保存RP/Profile后重新获取正式SP题组再提交。");
    return;
  }

  const choices = collectChoices();
  const required = currentAssignment.tasks.length;
  if (required === 0) {
    alert("当前没有可提交的题目，请先调整条件并重新获取SP题组。");
    return;
  }
  if (Object.keys(choices).length < required) {
    alert("请完成所有题目的选择。\n");
    return;
  }

  const respondentId = getOrCreateRespondentId();
  const payload = {
    assignment_id: currentAssignment.assignment_id,
    respondent_id: respondentId,
    design_save_name: designMeta.save_name || null,
    choices,
    tasks: currentAssignment.tasks || [],
  };
  const res = await fetch("/api/survey/sp-submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  const submitOk = !!(res.ok && !data.error);
  if (!submitOk) {
    alert(data.error || "保存SP答案失败");
    return;
  }

  localStorage.setItem(
    "survey_sp_last_submission",
    JSON.stringify({
      saved_at: new Date().toISOString(),
      respondent_id: respondentId,
      assignment_id: currentAssignment.assignment_id,
      choices,
      response: data,
    }),
  );

  const action = await showActionDialog(
    "SP答案保存成功",
    `提交ID: ${data.submission_id || currentAssignment.assignment_id}`,
    [
      { id: "confirm_save", label: "确定保存" },
      { id: "back", label: "返回页面" },
      { id: "go_profile", label: "返回个人/家庭信息页" },
      { id: "go_trip", label: "返回TripDiary页面" },
    ],
  );

  if (action === "confirm_save") {
    clearAllSurveyCache();
    localStorage.setItem(respondentKey, genRespondentId());
    window.location.href = isFileMode ? "profile.html" : "/survey/profile";
    return;
  }
  if (action === "go_profile") {
    window.location.href = isFileMode ? "profile.html" : "/survey/profile";
    return;
  }
  if (action === "go_trip") {
    window.location.href = isFileMode ? "trip_diary.html" : "/survey/trip-diary";
    return;
  }

  currentAssignment = null;
  localStorage.removeItem("survey_sp_current_assignment");
  document.getElementById("spSection").classList.add("hidden");
  document.getElementById("tasksContainer").innerHTML = "";
  updateSpInheritBanner();
  await fetchMetrics();
}

const loadBtn = document.getElementById("loadDesignBtn");
const submitBtn = document.getElementById("submitBtn");
const exportBtn = document.getElementById("exportRespondentJsonBtn");
if (loadBtn) loadBtn.addEventListener("click", loadDesign);
if (submitBtn) submitBtn.addEventListener("click", submitResponse);
if (exportBtn) exportBtn.addEventListener("click", exportCurrentRespondentJson);

(async function initApp() {
  createRpFields();
  initRespondentInheritance();
  await loadDesignOptions();
  updateSpInheritBanner();
  await fetchMetrics();
})();
