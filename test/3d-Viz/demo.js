import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const sceneMount = document.getElementById('sceneMount');
const diagramCanvas = document.getElementById('diagramCanvas');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const playTripBtn = document.getElementById('playTripBtn');
const playSpBtn = document.getElementById('playSpBtn');
const playAllBtn = document.getElementById('playAllBtn');
const playTrainBtn = document.getElementById('playTrainBtn');
const focusArchitectureBtn = document.getElementById('focusArchitectureBtn');
const focusPapersBtn = document.getElementById('focusPapersBtn');
const resetBtn = document.getElementById('resetBtn');
const toggleRotateBtn = document.getElementById('toggleRotateBtn');
const encoderTabBtn = document.getElementById('encoderTabBtn');
const decoderTabBtn = document.getElementById('decoderTabBtn');
const moduleButtons = Array.from(document.querySelectorAll('.module-btn'));
const diagramCtx = diagramCanvas ? diagramCanvas.getContext('2d') : null;
const flowModeButtons = [playTripBtn, playSpBtn, playTrainBtn, playAllBtn].filter(Boolean);

const statsRefs = {
  overallProgressValue: document.getElementById('overallProgressValue'),
  overallProgressFill: document.getElementById('overallProgressFill'),
  focusValue: document.getElementById('focusValue'),
  stageValue: document.getElementById('stageValue'),
  stackCountValue: document.getElementById('stackCountValue'),
  filledCountValue: document.getElementById('filledCountValue'),
  filledBreakdownValue: document.getElementById('filledBreakdownValue'),
  respondentValue: document.getElementById('respondentValue'),
  odValue: document.getElementById('odValue'),
  durationValue: document.getElementById('durationValue'),
  modeValue: document.getElementById('modeValue'),
  flagsValue: document.getElementById('flagsValue'),
  trainingPhaseValue: document.getElementById('trainingPhaseValue'),
  pretrainEpochValue: document.getElementById('pretrainEpochValue'),
  onlineUpdateValue: document.getElementById('onlineUpdateValue'),
  weightVersionValue: document.getElementById('weightVersionValue'),
  analysisValue: document.getElementById('analysisValue'),
  dimensionModuleValue: document.getElementById('dimensionModuleValue'),
  dimensionStageValue: document.getElementById('dimensionStageValue'),
  dimensionSourceValue: document.getElementById('dimensionSourceValue'),
  dimensionBeforeValue: document.getElementById('dimensionBeforeValue'),
  dimensionAfterValue: document.getElementById('dimensionAfterValue'),
  dimensionDetailValue: document.getElementById('dimensionDetailValue'),
  head1Value: document.getElementById('head1Value'),
  head2Value: document.getElementById('head2Value'),
  head3Value: document.getElementById('head3Value'),
  head4Value: document.getElementById('head4Value'),
  scenarioTagValue: document.getElementById('scenarioTagValue'),
  planValue: document.getElementById('planValue'),
  reasonCountValue: document.getElementById('reasonCountValue'),
  reasonListValue: document.getElementById('reasonListValue'),
  spProgressValue: document.getElementById('spProgressValue'),
  spProgressFill: document.getElementById('spProgressFill'),
  flowSummaryValue: document.getElementById('flowSummaryValue'),
};

const leftDocRefs = {
  title: document.getElementById('docModuleTitle'),
  source: document.getElementById('docSource'),
  content: document.getElementById('docSnippet'),
};

const dashboardState = {
  focusId: 'papers',
  focusLabel: '双页概览',
  stage: '初始化',
  trainingPhase: 'Idle',
  tripStacks: 3,
  spStacks: 3,
  pretrainEpoch: 0,
  onlineUpdate: 0,
  weightVersion: 'w0',
  analysisNote: '尚未开始',
  scenarioTag: 'G0',
  headOutputs: [0.25, 0.25, 0.25, 0.25],
  flowSummary: '从 respondent 的 RP / Trip 输入开始，依次查看 embedding、concat、多头注意力、loss 计算、问卷生成与下一位 respondent 的循环。',
  dimensionInfo: {
    module: '双页概览',
    stage: '初始化',
    before: '尚未开始',
    after: '点击模块或运行训练流程后显示',
    detail: '维度变化会随模块高亮与训练流程同步刷新。',
  },
};

const DOC_SOURCE_BY_FOCUS = {
  papers: 'docs/CURRENT_ALGORITHMS.md · templates/docs.html',
  architecture: 'docs/CURRENT_ALGORITHMS.md · docs/SELFATTENTION.md · templates/docs.html',
  tripPaper: 'docs/CURRENT_ALGORITHMS.md（RP/Trip 流程） · templates/docs.html',
  spPaper: 'docs/CURRENT_ALGORITHMS.md（SP 发放与回写） · templates/docs.html',
  rpPart: 'docs/CURRENT_ALGORITHMS.md（RP 输入拆分） · docs/SELFATTENTION.md · templates/docs.html',
  spPart: 'docs/CURRENT_ALGORITHMS.md（SP 上下文与变量槽位） · docs/SELFATTENTION.md · templates/docs.html',
  x_rp: 'docs/SELFATTENTION.md（输入张量 X_rp 定义） · templates/docs.html',
  x_spCtx: 'docs/SELFATTENTION.md（输入张量 X_hist 定义） · templates/docs.html',
  x_env: 'docs/SELFATTENTION.md（输入张量 X_env 定义） · templates/docs.html',
  x_cand: 'docs/SELFATTENTION.md（输入张量 X_cand 定义） · templates/docs.html',
};

const DIMENSION_STATES = {
  papers: {
    label: '双页概览',
    states: {
      default: {
        panelStage: '初始化',
        before: 'Trip Diary + SP Survey 页面\n尚未触发编码或训练流程',
        after: '等待 respondent 填写 RP / Trip / SP 字段',
        detail: '当前只是问卷载体层。真正进入模型前，页面字段会先整理为 X_rp、X_env、X_hist 和 X_cand；其中 X_hist 表示前 n-1 份已完成问卷提炼出的历史 SP 上下文。',
      },
    },
  },
  architecture: {
    label: '算法架构总览',
    states: {
      default: {
        panelStage: '总览',
        before: '输入: X_rp [B,d_rp], X_env [B,d_env],\nX_hist [B,L_hist,d_hist], X_cand [B,L_cand,d_cand]',
        after: '输出: H_enc [B,L_enc,d_model] -> Q_slot [B,T_max,d_model]\n-> count/mask/value/score heads -> checkpoint',
        detail: '这条链路对应文档里的 encoder-decoder 主流程：先编码 respondent、已收集历史上下文与候选条件，再并行生成整份 SP block；对 respondent 端，问卷始终是一次性发放和一次性填写。',
      },
    },
  },
  tripPaper: {
    label: 'Trip / RP 输入',
    states: {
      default: {
        panelStage: '原始表单',
        before: '问卷字段: 日期 / 成员 / OD / 出发到达时间 / 方式',
        after: '尚未编码为模型输入',
        detail: 'Trip Diary 页面负责收集 respondent 的 RP 与行程基础信息，后续会被整理成 X_rp 和部分 X_env 特征。',
      },
      filled: {
        panelStage: 'RP 编码准备',
        before: '原始表单字段 -> 清洗 / 归并 / 离散化',
        after: 'X_rp [B,d_rp]\nX_env [B,d_env]',
        detail: '填写完成后，这些字段会被压成 respondent 向量、环境向量和历史问卷摘要；decoder 端拿到的是并行 question queries，而 X_hist 来自此前已完成问卷的历史 SP 上下文。',
      },
    },
  },
  rpPart: {
    label: 'RP part',
    states: {
      default: {
        panelStage: 'RP 输入分组',
        before: 'Trip Diary / Profile 收集到的 respondent、家庭与行程基础字段',
        after: '优先进入 X_rp，也可派生部分 X_env 特征',
        detail: 'RP part 是原始 respondent 信息的入口层。这里还没进入 attention，只是把原始表单字段整理成可编码的特征集合。',
      },
    },
  },
  spPart: {
    label: 'SP part',
    states: {
      default: {
        panelStage: 'SP 上下文分组',
        before: '设计配置、候选变量、已完成 blocks 的历史题组摘要',
        after: '进入 X_hist / X_cand，并与 respondent 条件一起送入模型',
        detail: 'SP part 对应题目设计侧的信息入口，包括历史已发放 block 的统计摘要、候选变量模板、候选槽位和约束过滤后的上下文。',
      },
    },
  },
  x_rp: {
    label: 'X_rp 输入特征',
    states: {
      default: {
        panelStage: 'RP 特征整理',
        before: '原始变量（SELFATTENTION.md 3.1）：\n- gender / age_group / education / zone_id\n- household 与 profile 派生指标',
        after: 'X_rp ∈ \\mathbb{R}^{B\\times d_{rp}}\n建议编码：one-hot + ordinal + 归一化连续变量',
        detail: '计算口径：先做字段清洗和分段映射，再按变量类型编码（类别 one-hot、有序变量保序编码、连续变量标准化）。最终按 respondent 维拼接成 d_rp。',
      },
    },
  },
  x_spCtx: {
    label: 'X_hist 历史 SP 上下文',
    states: {
      default: {
        panelStage: 'SP 历史上下文',
        before: '历史上下文（SELFATTENTION.md 3.3）：\n- 前 n-1 份已完成问卷的题组统计\n- 变量激活统计 rho_v\n- 变量值统计 mean/std\n- expected utility spread',
        after: 'X_hist ∈ \\mathbb{R}^{B\\times L_{hist}\\times d_{hist}}\n由历史题组 token + summary token 组成',
        detail: '计算口径：对前 n-1 份已完成问卷中的历史题 token 累计 m_{j,v} 与 x_{j,v}，得到激活频率 rho_v、条件均值/方差和 spread，再写入上下文 token 序列。它不表示当前 respondent 这一个 block 内尚未提交的实时作答。',
      },
    },
  },
  x_env: {
    label: 'X_env 环境特征',
    states: {
      default: {
        panelStage: '环境统计特征',
        before: '环境特征（SELFATTENTION.md 3.2）：\n- 样本进度 n/N_target\n- zone 偏差与覆盖率\n- design 分发次数\n- 参数稳定度与 D-error 水平',
        after: 'X_env ∈ \\mathbb{R}^{B\\times d_{env}}\n批次内可广播或按 respondent 复制',
        detail: '计算口径：实时聚合采集进度、分区偏差、拟合稳定性指标，做归一化后拼接为 d_env，用于描述“当前系统状态”。',
      },
    },
  },
  x_cand: {
    label: 'X_cand 候选变量模板',
    states: {
      default: {
        panelStage: '候选集合输入',
        before: '候选模板（SELFATTENTION.md 3.4）：\n- 变量集合与上下界 [l_v,u_v]\n- 逻辑约束/主导性过滤\n- alternative/group 结构标签',
        after: 'X_cand ∈ \\mathbb{R}^{B\\times L_{cand}\\times d_{cand}}\n每个 token 对应一个候选变量槽位',
        detail: '计算口径：先做可行性过滤（边界、约束、主导性），再把保留模板编码成 token 序列，供 encoder 与 respondent 状态联合建模。',
      },
    },
  },
  inputEmbedding: {
    label: 'Input Embedding',
    states: {
      default: {
        box: 'X_{rp}, X_{hist}, X_{env}, X_{cand}\n→ E_{rp}, E_{hist}, E_{env}, E_{cand}',
        panelStage: '输入投影',
        before: '输入变量 / 输入张量\nX_{rp} \\in \\mathbb{R}^{B\\times d_{rp}}\nX_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{hist}}\nX_{env} \\in \\mathbb{R}^{B\\times d_{env}}\nX_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{cand}}',
        after: '输出张量 / 编码结果\nE_{rp} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}\nE_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{model}}\nE_{env} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}\nE_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{model}}',
        detail: '为什么是 1+L_{hist}+1+L_{cand}：\nE_{rp} 只对应 1 个 respondent token，E_{hist} 对应 L_{hist} 个历史上下文 token，E_{env} 只对应 1 个环境 token，E_{cand} 对应 L_{cand} 个候选 token。\n所以拼接后序列长度 L_{enc}=1+L_{hist}+1+L_{cand}。\n注：这一“汇聚/拼接”通常是 concat 操作本身，不新增可训练参数（3D 中汇聚节点仅用于结构示意）。',
      },
      project: {
        box: 'Embed_*: d_* -> d_model\nE_rp/E_env/E_hist/E_cand',
        panelStage: '输入投影进行中',
        before: '原始维度不一致:\nd_rp / d_env / d_hist / d_cand',
        after: '统一到 d_model:\n[B,1,d_model] 或 [B,L,d_model]',
        detail: '这里不改变 batch B，只映射最后一维到 d_{model}。\n其中 RP 和 Env 都是单 token（长度=1），Ctx 和 Cand 是多 token（长度分别是 L_{hist}, L_{cand}）。',
      },
    },
  },
  positionalEncoding: {
    label: 'Positional Encoding',
    states: {
      default: {
        box: 'E_hist/E_cand + PE\n[B,L,d_model]\n长度保持不变',
        panelStage: '位置编码',
        before: 'E_hist [B,L_hist,d_model]\nE_cand [B,L_cand,d_model]',
        after: '加位置编码后仍为\n[B,L_hist,d_model] / [B,L_cand,d_model]',
        detail: '位置编码不改变张量形状，只给序列里每个 token 注入“第几个位置”的信息，避免模型只看到内容看不到顺序。',
      },
      inject: {
        box: 'Add positional code\nshape invariant\n[B,L,d_model]',
        panelStage: '位置注入中',
        before: '当前 token 只有内容 embedding',
        after: 'token 内容 + 位置索引共同进入 encoder',
        detail: '这一层最重要的是把不同来源的条件输入映射到统一维度，尤其让 X_hist 与 X_cand 的统计摘要能够进入同一个 encoder 记忆空间。',
      },
    },
  },
  encoder: {
    label: 'Encoder',
    states: {
      default: {
        box: 'X_enc [B,L_enc,d_model]\nSelf-Attn + FFN\n-> H_enc [B,L_enc,d_model]',
        panelStage: '上下文编码',
        before: 'X_enc = concat(E_rp,E_hist,E_env,E_cand)\nL_enc = 1 + L_hist + 1 + L_cand',
        after: 'H_enc [B,L_enc,d_model]',
        detail: 'Encoder 把 respondent、环境、历史 SP 上下文和候选变量模板一起编码为统一记忆。Self-attention 内部先用 W_Q、W_K、W_V 把 X_enc 投影成 Q/K/V ∈ R^{B×L_enc×(h·d_h)}，再 reshape 为 R^{B×h×L_enc×d_h}。单个 head 内计算 S_i=Q_iK_i^T/sqrt(d_h)、A_i=softmax(S_i)、O_i=A_iV_i，最后把 h 个 head 的 O_i concat 并乘 W_O 得到 H_enc。',
      },
      concat: {
        box: 'concat(E_rp,E_env,\nE_hist,E_cand)\nL_enc=1+1+L_hist+L_cand',
        panelStage: '序列拼接',
        before: '四组 token 分散存放',
        after: '单一 encoder 序列\nX_enc [B,L_enc,d_model]',
        detail: '先拼成长序列，再做 self-attention；这样 encoder 能跨 respondent、环境、历史题目和候选模板看全局依赖。',
      },
      attn: {
        box: 'Q/K/V_e [B,h,L_enc,d_h]\nS_e [B,h,L_enc,L_enc]\nO_e -> H_enc',
        panelStage: '自注意力计算',
        before: 'X_enc [B,L_enc,d_model]',
        after: 'Q_e, K_e, V_e [B,h,L_enc,d_h]\nA_e [B,h,L_enc,L_enc]\nH_enc [B,L_enc,d_model]',
        detail: '工程实现里常把所有 head 的权重合成大矩阵：W_Q/W_K/W_V ∈ R^{d_model×(h·d_h)}。输入 X_enc ∈ R^{B×L_enc×d_model}，先得到 Q/K/V ∈ R^{B×L_enc×(h·d_h)}，再 reshape/transposed 为 R^{B×h×L_enc×d_h}。单个 head 的 S_i/A_i ∈ R^{B×L_enc×L_enc}，O_i ∈ R^{B×L_enc×d_h}；多个 O_i 沿最后一维拼接为 R^{B×L_enc×(h·d_h)}，再经 W_O 回到 H_enc ∈ R^{B×L_enc×d_model}。',
      },
    },
  },
  respondentTargetHead: {
    label: 'Respondent Target Head',
    states: {
      default: {
        box: 'respondent_target_head\n[B,C_sample]\n-> sampling recommendation',
        panelStage: '定向采样建议',
        before: 'H_enc pooled -> g_enc [B,d_model]',
        after: 'p_sample [B,C_sample]\ncell priority + needed_n',
        detail: '这一支从 encoder 的全局状态输出剩余样本建议，例如 zone、gender、age、education 等 RP cell 的优先级。它不生成 SP 题，也不强制改变样本，只给调查员或 dashboard 提供覆盖缺口参考。',
      },
    },
  },
  outputEmbedding: {
    label: 'Embedding&Concat',
    states: {
      default: {
        box: 'parallel question queries\nQ_slot [B,T_max,d_model]',
        panelStage: '解码器输入',
        before: '当前 respondent 的并行题位 queries',
        after: 'Q_slot [B,T_max,d_model]',
        detail: 'Decoder 输入不是 RP 原始特征，也不是 respondent 已填写答案，而是一组并行 question queries。它们表示一整份问卷中待生成的题位，占位后再统一与 H_enc 做条件融合。',
      },
      tokens: {
        box: 'parallel queries\n-> Q_slot [B,T_max,d_model]',
        panelStage: '并行题位初始化',
        before: 'question query slots',
        after: '并行题位送入 decoder',
        detail: '这里一次性放入 T_max 个并行题位 queries。它们没有页面顺序语义；题数由 count head 决定，哪些候选题进入 block 由 slot_select_head 控制。',
      },
    },
  },
  questionSetAttention: {
    label: 'Question-Set Attention',
    states: {
      default: {
        box: 'Q_s/K_s/V_s [B,h,T_max,d_h]\nS_s [B,h,T_max,T_max]\nfull attention',
        panelStage: '题位自注意力',
        before: 'Q_slot [B,T_max,d_model]',
        after: '并行题位的关联表示 H_slot',
        detail: 'Question-set self-attention 的 Q/K/V 都来自并行 question queries。先通过 W_Q/W_K/W_V 把 Q_slot ∈ R^{B×T_max×d_model} 投影到 Q/K/V ∈ R^{B×T_max×(h·d_h)}，再 reshape 为 R^{B×h×T_max×d_h}。随后在 S_i ∈ R^{B×T_max×T_max} 上做 full self-attention，建模整份问卷内部候选题之间的关系，不做 causal mask。',
      },
      masked: {
        box: 'all-to-all slot relation\nA_s over all slots\nshape kept',
        panelStage: '题位关系矩阵',
        before: 'S_s [B,h,T_max,T_max]',
        after: 'A_s [B,h,T_max,T_max]\n题位间关系权重',
        detail: '这里保留 full self-attention，形状仍为 [B,h,T_max,T_max]。它建模的是整份问卷中各题位之间的互补、重复和覆盖关系，而不是题内时序关系。',
      },
    },
  },
  decoder: {
    label: 'Decoder',
    states: {
      default: {
        box: 'Cross-Attn + FFN\nO_cross [B,T_max,d_model]\nH_slot [B,T_max,d_model]',
        panelStage: '条件生成',
        before: '并行题位表示 + encoder 记忆 H_enc',
        after: '各题位隐藏状态 H_slot [B,T_max,d_model]',
        detail: 'Decoder 先做并行 question-set self-attention，再对 H_enc 做 cross-attention，最后输出每个题位的隐藏表示 H_slot。',
      },
      cross: {
        box: 'Q_cross [B,h,T_max,d_h]\nK/V_cross [B,h,L_enc,d_h]\nS_cross [B,h,T_max,L_enc]',
        panelStage: '跨注意力读取记忆',
        before: 'decoder query + H_enc',
        after: 'O_cross [B,T_max,d_model]\n融合 respondent / env / hist / cand',
        detail: 'Cross-attention 让每个并行题位在生成时按需读取 encoder 里的 respondent、环境、历史问卷和候选变量信息。',
      },
      refine: {
        box: 'updated H_enc + Q_slot\n-> refined H_slot\n[B,T_max,d_model]',
        panelStage: '更新后重解码',
        before: 'checkpoint 更新后的权重',
        after: '新一轮 block 生成隐藏状态',
        detail: '在线更新完成后，decoder 会用新权重重新生成整份 block，因此相同 respondent 输入下，题数、题间结构和变量值都可能重新调整。',
      },
    },
  },
  multiHeads: {
    label: 'Multi-Head Outputs',
    states: {
      default: {
        box: 'count [B,K_count]\nmask/value [B,T_max,V]\nscore [B,T_max,1]',
        panelStage: '多头输出',
        before: 'H_slot [B,T_max,d_model]',
        after: 'count_logits [B,K_count]\nmask/value [B,T_max,V]\nscore [B,T_max,1]',
        detail: '这里把并行题位表示拆成四类输出：本次 block 的题数、每题变量是否激活、变量值取什么、每题质量如何。',
      },
      heads: {
        box: '4 heads from H_slot\ncount + structure\nvalue + quality',
        panelStage: '结构与数值同步生成',
        before: 'decoder 并行题位隐藏状态',
        after: '题数 / 变量激活情况 / 变量值 / 题目得分',
        detail: 'Multi-head 允许“题数可变、变量可有可无、值可连续”同时成立，而不是只做一个离散 softmax 选题。',
      },
    },
  },
  analysis: {
    label: 'Analysis',
    states: {
      default: {
        box: 'answers [B,T_q]\n+ reason / notes / logs\n-> batch stats',
        panelStage: '反馈分析',
        before: 'respondent 的选择结果与补充说明',
        after: '激活频次 / 变量值统计 / expected utility spread / adjusted pseudo R^2',
        detail: '分析模块把作答结果聚合成统计指标，既服务 reward，也服务你在文档里关注的变量激活情况、变量值分布和敏感性分析。',
      },
      feedback: {
        box: 'answers -> metrics\ncoverage / spread\nstability / fit',
        panelStage: '在线评估',
        before: 'question block + respondent decisions',
        after: 'batch-level metric vector / dashboard 指标',
        detail: '这一步不一定只输出一个张量，更重要的是把当前批次表现整理成可用于 checkpoint 更新和页面统计的指标集合。',
      },
    },
  },
  weights: {
    label: 'Weights / Checkpoint',
    states: {
      default: {
        box: 'theta_k / checkpoint\nw_k -> w_{k+1}',
        panelStage: '权重存储',
        before: 'head 输出或 analysis 指标',
        after: '保存为新的 checkpoint / 参数版本',
        detail: 'Weights 模块代表当前模型参数状态。它既接预训练阶段的静态更新，也接在线阶段的增量更新。',
      },
      checkpoint: {
        box: 'save checkpoint\nversion += 1\nweights persisted',
        panelStage: '保存权重',
        before: '本轮 head 输出与损失反向传播结果',
        after: 'w_k 写入磁盘 / 内存状态',
        detail: '预训练阶段重点是把 warmup 得到的参数固定下来，形成后续题面生成要读取的 checkpoint。',
      },
      update: {
        box: 'online update\nmetrics -> grad / reward\ncheckpoint refresh',
        panelStage: '在线更新',
        before: 'analysis 返回的统计与拟合指标',
        after: 'w_{k+1} / theta_{k+1}',
        detail: '收到 respondent 真实作答后，系统会用新的统计信息修正参数，再触发下一轮 decoder 生成。',
      },
    },
  },
  spPaper: {
    label: 'SP Survey',
    states: {
      default: {
        panelStage: '待生成题面',
        before: 'decoder 尚未把 block 写回问卷页',
        after: '等待 G1 / G2 等情景方案被渲染到页面',
        detail: 'SP 页只是展示层。真正的 block 结构来自 decoder 与 multi-head 输出，不是页面本身静态定义出来的。',
      },
      generated: {
        panelStage: '题面已生成',
        before: 'count/mask/value/score heads 输出完成',
        after: 'question block [B,T_q,V]\n写回 SP 页面展示',
        detail: '当 weights -> decoder -> heads 路径跑完后，SP 页面会一次性收到一整组 block 题目和对应变量值，随后 respondent 再整份填写。',
      },
      answered: {
        panelStage: '作答已回收',
        before: 'question block [B,T_q,V]',
        after: 'answers [B,T_q]\nreason / note -> analysis',
        detail: '受访者完成作答后，选择结果、勾选原因和文本说明会一起回流到 analysis 模块，用于拟合和在线更新。',
      },
    },
  },
};

Object.assign(DIMENSION_STATES, {
  encoderAddNormBottom: {
    label: '编码器 Add & Norm',
    states: {
      default: {
        panelStage: '残差归一化',
        before: 'Self-Attn 输出 + residual\n[B,L_enc,d_model]',
        after: 'LayerNorm 后仍为\n[B,L_enc,d_model]',
        detail: 'Add & Norm 不改变张量形状，只把子层输出与残差相加，再做 LayerNorm，稳定训练并保留原始信息通路。',
        box: 'Add & Norm\n[B,L_enc,d_model]',
      },
    },
  },
  encoderFeedForward: {
    label: '编码器前馈网络层',
    states: {
      default: {
        panelStage: '前馈变换',
        before: 'H [B,L_enc,d_model]',
        after: 'FFN(H) [B,L_enc,d_model]',
        detail: '前馈网络逐 token 独立工作，通常是 d_model -> d_ff -> d_model，提升非线性表达能力，但输出形状保持不变。',
        box: '前馈网络层\n[B,L_enc,d_model]',
      },
    },
  },
  encoderAddNormTop: {
    label: '编码器 Add & Norm',
    states: {
      default: {
        panelStage: '输出归一化',
        before: 'FFN 输出 + residual\n[B,L_enc,d_model]',
        after: 'H_enc [B,L_enc,d_model]',
        detail: '这是 encoder block 的第二个残差归一化出口，输出就是后续 decoder cross-attention 要读取的上下文记忆 H_enc。',
        box: 'Add & Norm\nH_enc [B,L_enc,d_model]',
      },
    },
  },
  decoderAddNormSelf: {
    label: '解码器 Add & Norm',
    states: {
      default: {
        panelStage: '自注意力后归一化',
        before: 'question-set self-attn 输出 + residual\n[B,T_max,d_model]',
        after: '归一化后仍为\n[B,T_max,d_model]',
        detail: '集合自注意力后的 Add & Norm 稳定候选题集合表示，再把结果送入交叉注意力读取 encoder 记忆。',
        box: 'Add & Norm\n[B,T_max,d_model]',
      },
    },
  },
  decoderCrossAttention: {
    label: '多头交叉注意力层',
    states: {
      default: {
        panelStage: '读取编码器记忆',
        before: 'Q_cross [B,h,T_max,d_h]\nK/V_cross [B,h,L_enc,d_h]',
        after: 'O_cross [B,T_max,d_model]',
        detail: 'Cross-attention 与 self-attention 的差别在输入来源：Q=H_dec W_Q，K=H_enc W_K，V=H_enc W_V。投影后 reshape 为 Q ∈ R^{B×h×T_max×d_h}，K/V ∈ R^{B×h×L_enc×d_h}；单个 head 的 S_i/A_i ∈ R^{B×T_max×L_enc}。多个 head 的 O_i concat 后乘 W_O，回到 O_cross ∈ R^{B×T_max×d_model}。',
        box: 'Cross-Attn\nO_cross [B,T_max,d_model]',
      },
    },
  },
  decoderAddNormCross: {
    label: '解码器 Add & Norm',
    states: {
      default: {
        panelStage: '交叉注意力后归一化',
        before: 'cross-attn 输出 + residual\n[B,T_max,d_model]',
        after: '归一化后仍为\n[B,T_max,d_model]',
        detail: '这个 Add & Norm 位于 cross-attention 之后，用于把 encoder 条件信息和 decoder 历史信息稳定融合。',
        box: 'Add & Norm\n[B,T_max,d_model]',
      },
    },
  },
  decoderAddNormTop: {
    label: '解码器 Add & Norm',
    states: {
      default: {
        panelStage: '输出归一化',
        before: 'FFN 输出 + residual\n[B,T_max,d_model]',
        after: '解码器输出表示\n[B,T_max,d_model]',
        detail: '这是 decoder block 最后的归一化出口，后面接 count、slot_select、mask、value、score 等多个 head。',
        box: 'Add & Norm\n[B,T_max,d_model]',
      },
    },
  },
  linearHead: {
    label: '线性层',
    states: {
      default: {
        panelStage: 'logit 投影',
        before: 'decoder 输出 [B,T_max,d_model]\n或单步 h_t [B,d_model]',
        after: 'logits [B,V]\n或 [B,T_max,V]',
        detail: '线性层把 d_model 投影到动作 / 词表 / 变量空间，对应输出层前的未归一化分数。',
        box: '线性层\nlogits [B,V]',
      },
    },
  },
  softmaxHead: {
    label: 'Softmax层',
    states: {
      default: {
        panelStage: '概率归一化',
        before: 'logits [B,V]\n或 [B,T_max,V]',
        after: 'prob [B,V]\n或 [B,T_max,V]',
        detail: 'Softmax 把线性层输出转成概率分布。若是分类输出，这一步对应每个 action / token 的选择概率。',
        box: 'Softmax\nprob [B,V]',
      },
    },
  },
  concatPanel: {
    label: 'Token Concat',
    states: {
      default: {
        panelStage: '序列拼接',
        before: 'E_rp [B,1,d_model]\nE_hist [B,L_hist,d_model]\nE_env [B,1,d_model]\nE_cand [B,L_cand,d_model]',
        after: 'X_{enc}^{(viz)}\n[B,12,6],\\ L=1+5+1+5',
        detail: 'Concat 可视化把四组 embedding 重新排布为一个 L×d 矩阵：这里固定 L=1+5+1+5=12，d=6。颜色表示该 L 段来自哪个 embedding 分支；整张矩阵沿 y 方向向下层叠表示 batch 维 B。',
        box: 'concat\nX_{enc}^{(viz)} [B,12,6]',
      },
    },
  },
  batchAttention: {
    label: '多样本自注意力',
    states: {
      default: {
        panelStage: '批量 attention',
        before: 'X_{enc}^{(viz)} [B,12,6]\nsplit_L -> h × [B,L/h,6]',
        after: 'Q/K/V, S=QK^T/sqrt(d), A=softmax(S), O=AV\nconcat(O) -> H_{enc}^{(viz)} [B,12,6]',
        detail: '这一阶段强调 B>1 的情况：多位 respondent 或多份 block 会同时进入 encoder。当前可视化口径下，attention 输入和输出都使用 L=12、d=6；随后按 L 方向切成 3 个 head，每个 head 处理 4 个位置。',
        box: 'Q/K/V -> split_L -> S -> A -> O\nB,h,L/h,d',
      },
    },
  },
  maskHead: {
    label: 'Mask Head',
    states: {
      default: {
        panelStage: '变量激活决策',
        before: 'H_slot [B,T_max,d_model]',
        after: 'mask_logits [B,T_max,V]\nmask_prob [B,T_max,V]\nmask ∈ {0,1}^{B,T_max,V}',
        detail: 'Mask head 为整份问卷中的每个题位同时输出变量激活向量。它决定的是每道题内部哪些变量出现，是并行 block 生成里的“结构 head”。',
        box: 'mask_head\n[B,T_max,V]',
      },
    },
  },
  valueHead: {
    label: 'Value Head',
    states: {
      default: {
        panelStage: '变量值生成',
        before: 'H_slot [B,T_max,d_model]',
        after: 'raw_value [B,T_max,V]\nvalue [B,T_max,V]',
        detail: 'Value head 在 mask head 给出的结构条件下，为已激活变量生成连续值或映射后的离散值，因此题目不再局限于固定 attribute levels 的静态挑选。',
        box: 'value_head\n[B,T_max,V]',
      },
    },
  },
  scoreHead: {
    label: 'Score Head',
    states: {
      default: {
        panelStage: '题目质量评分',
        before: 'H_slot [B,T_max,d_model]',
        after: 'score [B,T_max,1]',
        detail: 'Score head 输出当前题或当前 block 的辅助质量分数，可理解为信息性、排序优先级或预期有效性的代理量。',
        box: 'score_head\n[B,T_max,1]',
      },
    },
  },
  countHead: {
    label: 'Count Head',
    states: {
      default: {
        panelStage: '题数判定',
        before: 'H_slot [B,T_max,d_model]',
        after: 'count_logits [B,K_count]\np_count [B,K_count]',
        detail: 'Count head 先决定当前 respondent 这一份问卷生成多少题。随后由 slot_select_head 选择有效题位，因此题数是整份问卷级别的一次性判定，而不是逐题终止判定。',
        box: 'count_head\n[B,K_count]',
      },
    },
  },
  slotSelectHead: {
    label: 'Slot Select Head',
    states: {
      default: {
        panelStage: '候选题集合选择',
        before: 'H_slot [B,T_max,d_model] + count_head 输出 T_q',
        after: 'slot_logits [B,T_max]\nM_slot ∈ {0,1}^{B,T_max}',
        detail: 'Slot select head 对所有候选题位打分，并在 count_head 给出题数后选出 top-k 个进入最终 block。它不是 causal mask，也不是默认前 k 个题位有效。',
        box: 'slot_select_head\n[B,T_max]',
      },
    },
  },
  outputBlock: {
    label: 'Question Block',
    states: {
      default: {
        panelStage: '题组生成',
        before: 'count/slot_select/mask/value/score 多头输出',
        after: 'question block [B,T_q,V]\nslot_select_head 决定哪些候选题进入 block',
        detail: '多头输出会被组装成真正分发给 respondent 的题组。count_head 先决定题数，再由 slot_select_head 选择有效候选题位，最后一次性写入整份问卷。',
        box: 'question block\n[B,T_q,V]',
      },
    },
  },
  lossPanel: {
    label: 'Loss 汇总',
    states: {
      default: {
        panelStage: '多头损失计算',
        before: 'L_count + L_slot + L_mask + L_value + L_score',
        after: 'L_total = λ_c L_count + λ_s L_slot + λ_m L_mask + λ_v L_value + λ_q L_score',
        detail: '训练阶段会分别计算题数、slot 选择、变量激活、变量取值和题目质量五类损失，再按权重系数汇总成总损失。这里也可附加约束惩罚、边界惩罚或 adjusted McFadden pseudo R^2 奖励项。',
        box: 'L_total\nΣ λ_i L_i',
      },
    },
  },
  backprop: {
    label: 'Backprop / Update',
    states: {
      default: {
        panelStage: '反向传播与更新',
        before: '∂L_total/∂θ\n梯度已从 head 回传到 decoder / encoder',
        after: 'θ_k -> θ_{k+1}\ncheckpoint refresh',
        detail: '从多头损失开始，梯度会逆向流过 decoder、cross-attention、encoder 和 embedding。优化器执行一步更新后，新权重会写回 checkpoint。',
        box: 'backprop\nθ_k -> θ_{k+1}',
      },
    },
  },
  dispatchLoop: {
    label: '问卷分发闭环',
    states: {
      default: {
        panelStage: '生成-分发-回收-下一位',
        before: 'question block + count / slot_select 决策',
        after: 'SP 页面展示 -> respondent 作答 -> analysis -> 新 respondent RP',
        detail: '这是完整在线系统闭环：生成题组并分发，回收答案后写回分析模块，再进入下一位 respondent 的 RP / Trip 输入。',
        box: 'dispatch loop\nrespondent -> next',
      },
    },
  },
});

const SCENE_CENTER = new THREE.Vector3(60, 40, 0);
const PAPER_OVERVIEW_CAMERA = new THREE.Vector3(60, 230, 2300);
const ARCHITECTURE_CENTER = new THREE.Vector3(60, 180, 0);
const ARCHITECTURE_CAMERA = new THREE.Vector3(60, 320, 2100);

const webglScene = new THREE.Scene();
webglScene.background = new THREE.Color(0x2a0810);
webglScene.fog = new THREE.Fog(0x2a0810, 1800, 3600);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 1, 4200);
camera.position.copy(PAPER_OVERVIEW_CAMERA);

const webglRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
webglRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
webglRenderer.setSize(window.innerWidth, window.innerHeight);
webglRenderer.outputColorSpace = THREE.SRGBColorSpace;
webglRenderer.setClearColor(0x2a0810, 1);
sceneMount.appendChild(webglRenderer.domElement);

const controls = new OrbitControls(camera, webglRenderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enableRotate = true;
controls.enableZoom = true;
controls.minDistance = 0.1;
controls.maxDistance = 10000000;
controls.enablePan = true;
controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
controls.minPolarAngle = THREE.MathUtils.degToRad(28);
controls.maxPolarAngle = THREE.MathUtils.degToRad(84);
controls.minAzimuthAngle = -Math.PI * 0.62;
controls.maxAzimuthAngle = Math.PI * 0.62;
controls.target.copy(SCENE_CENTER);
controls.autoRotate = false;
controls.autoRotateSpeed = 0.55;
webglRenderer.domElement.addEventListener('contextmenu', (event) => {
  event.preventDefault();
});

const ambientLight = new THREE.HemisphereLight(0xd5e6ff, 0x1b2f49, 2.45);
webglScene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.05);
keyLight.position.set(60, 520, 1060);
webglScene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xfff4d6, 1.12);
rimLight.position.set(260, 340, 900);
webglScene.add(rimLight);

const fillLight = new THREE.PointLight(0xa8dbff, 0.85, 4200);
fillLight.position.set(-360, 260, 1200);
webglScene.add(fillLight);

const desk = new THREE.Mesh(
  new THREE.BoxGeometry(2800, 10, 2400),
  new THREE.MeshPhongMaterial({
    color: 0x0f1b2b,
    transparent: false,
    opacity: 1,
    shininess: 18,
  }),
);
desk.position.y = -280;
desk.visible = false;
webglScene.add(desk);

const deskGrid = new THREE.GridHelper(2200, 30, 0x2e5d87, 0x1a314a);
deskGrid.position.y = -272;
deskGrid.material.transparent = true;
deskGrid.material.opacity = 0.24;
deskGrid.visible = false;
webglScene.add(deskGrid);

const particleGeo = new THREE.BufferGeometry();
const particleCount = 420;
const particlePos = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount; i += 1) {
  particlePos[i * 3] = THREE.MathUtils.randFloatSpread(2200);
  particlePos[i * 3 + 1] = THREE.MathUtils.randFloat(-200, 520);
  particlePos[i * 3 + 2] = THREE.MathUtils.randFloatSpread(1800);
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
const particles = new THREE.Points(
  particleGeo,
  new THREE.PointsMaterial({
    color: 0x7dd3fc,
    size: 4.5,
    transparent: true,
    opacity: 0.42,
    sizeAttenuation: true,
  }),
);
particles.visible = false;
webglScene.add(particles);

const architectureGroup = new THREE.Group();
architectureGroup.scale.setScalar(1.12);
architectureGroup.position.y = 24;
webglScene.add(architectureGroup);

const moduleMap = new Map();
const connectorMap = new Map();
const focusTargets = new Map();
const moduleClickTargets = [];
const embeddingNeuronTargets = [];
const concatCubeTargets = [];
const attentionTensorTargets = [];
let hoveredEmbeddingNode = null;
let selectedEmbeddingNode = null;
let hoveredConcatCube = null;
let hoveredAttentionTensorNode = null;
const liveArrows = [];
const moduleAnimators = [];
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let routeAnimation = null;
let isPlaying = false;
let rotateEnabled = false;
let focusTween = null;

const embeddingTooltip = document.createElement('div');
embeddingTooltip.style.cssText = [
  'position:absolute',
  'display:none',
  'pointer-events:none',
  'z-index:60',
  'max-width:360px',
  'padding:8px 10px',
  'border-radius:8px',
  'font-family:monospace',
  'font-size:11px',
  'line-height:1.45',
  'white-space:normal',
  'color:#d1fae5',
  'background:rgba(0,0,0,0.82)',
  'border-left:3px solid #3a86ff',
  'box-shadow:0 6px 16px rgba(0,0,0,0.35)',
  'backdrop-filter:blur(6px)',
].join(';');
document.body.appendChild(embeddingTooltip);

function setStatus(badge, text) {
  statusBadge.textContent = '模块详情';
  statusText.textContent = text;
}

function setFlowSummary(text) {
  dashboardState.flowSummary = text;
  if (statsRefs.flowSummaryValue) {
    statsRefs.flowSummaryValue.textContent = text;
  }
}

function getDocSourceText(focusId = 'papers') {
  return DOC_SOURCE_BY_FOCUS[focusId]
    || DOC_SOURCE_BY_FOCUS.architecture;
}

function updateLeftDocPanel() {
  if (!leftDocRefs.title || !leftDocRefs.source || !leftDocRefs.content) return;
  leftDocRefs.title.textContent = `文档关联：${dashboardState.dimensionInfo.module}`;
  leftDocRefs.source.textContent = `来源：${getDocSourceText(dashboardState.focusId)}`;
  setMathBlock(
    leftDocRefs.content,
    `阶段：${dashboardState.dimensionInfo.stage}\n\n输入：\n${dashboardState.dimensionInfo.before}\n\n输出：\n${dashboardState.dimensionInfo.after}\n\n说明：\n${dashboardState.dimensionInfo.detail}`,
  );
}

function setFlowModeButton(activeBtn = null) {
  flowModeButtons.forEach((btn) => {
    btn.classList.remove('is-primary', 'is-warm');
    if (btn === activeBtn) {
      btn.classList.add('is-primary');
      if (btn === playTrainBtn) btn.classList.add('is-warm');
    }
  });
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function lerpColorHex(a, b, t) {
  const ca = new THREE.Color(a);
  const cb = new THREE.Color(b);
  return `#${ca.lerp(cb, t).getHexString()}`;
}

const CANVAS_MODULE_IDS = new Set([
  'x_rp',
  'x_spCtx',
  'x_env',
  'x_cand',
  'inputEmbedding',
  'concatPanel',
  'encoder',
  'batchAttention',
  'encoderAddNormBottom',
  'encoderFeedForward',
  'encoderAddNormTop',
  'outputEmbedding',
  'questionSetAttention',
  'decoderAddNormSelf',
  'decoderCrossAttention',
  'decoderAddNormCross',
  'decoder',
  'decoderAddNormTop',
  'maskHead',
  'valueHead',
  'scoreHead',
  'countHead',
  'slotSelectHead',
  'outputBlock',
  'lossPanel',
  'backprop',
  'dispatchLoop',
]);

const ENCODER_CANVAS_IDS = new Set([
  'x_rp',
  'x_spCtx',
  'x_env',
  'x_cand',
  'inputEmbedding',
  'concatPanel',
  'encoder',
  'encoderAddNormBottom',
  'encoderFeedForward',
  'encoderAddNormTop',
]);

const DECODER_CANVAS_IDS = new Set([
  'outputEmbedding',
  'questionSetAttention',
  'decoderAddNormSelf',
  'decoderCrossAttention',
  'decoderAddNormCross',
  'decoder',
  'decoderAddNormTop',
  'maskHead',
  'valueHead',
  'scoreHead',
  'countHead',
  'slotSelectHead',
  'outputBlock',
]);

const DIAGRAM_LINE_COLOR = 'rgba(248, 250, 252, 0.96)';
const DIAGRAM_LINE_FAINT = 'rgba(248, 250, 252, 0.38)';
const DIAGRAM_FONT_FAMILY = '"Avenir Next", "PingFang SC", "Helvetica Neue", Arial, sans-serif';
const DIAGRAM_TEXT_DARK = '#223047';
const DIAGRAM_FRAME_FILL = 'rgba(248, 243, 235, 0.94)';
const DIAGRAM_MODULE_COLORS = {
  addNorm: '#F3EDC9',
  attention: '#F1DDC1',
  feedForward: '#D9E7F1',
  embedding: '#F1E1DF',
  neutral: '#F7F2EA',
  head: '#EEE1C8',
  tagBlue: '#D9E8F7',
  tagLavender: '#E7E1F3',
};

const diagramState = {
  selectedId: '',
  hoverId: '',
  activeIds: new Set(),
  activeConnectorId: '',
  routeStarted: 0,
  routeDuration: 0,
  hitRegions: [],
};

function createRect(x, y, width, height) {
  return { x, y, width, height };
}

function rectCenter(rect) {
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
}

function rectTop(rect) {
  return { x: rect.x + rect.width / 2, y: rect.y };
}

function rectBottom(rect) {
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height };
}

function rectBottomAt(rect, ratio = 0.5) {
  return { x: rect.x + rect.width * ratio, y: rect.y + rect.height };
}

function rectLeft(rect, ratio = 0.5) {
  return { x: rect.x, y: rect.y + rect.height * ratio };
}

function rectRight(rect, ratio = 0.5) {
  return { x: rect.x + rect.width, y: rect.y + rect.height * ratio };
}

function drawRoundedRect(ctx, rect, radius, fill, stroke, lineWidth = 3, shadow = false) {
  const r = Math.min(radius, rect.width / 2, rect.height / 2);
  ctx.save();
  if (shadow) {
    ctx.shadowColor = 'rgba(15, 23, 42, 0.12)';
    ctx.shadowBlur = 18;
    ctx.shadowOffsetY = 10;
  }
  ctx.beginPath();
  ctx.moveTo(rect.x + r, rect.y);
  ctx.lineTo(rect.x + rect.width - r, rect.y);
  ctx.quadraticCurveTo(rect.x + rect.width, rect.y, rect.x + rect.width, rect.y + r);
  ctx.lineTo(rect.x + rect.width, rect.y + rect.height - r);
  ctx.quadraticCurveTo(rect.x + rect.width, rect.y + rect.height, rect.x + rect.width - r, rect.y + rect.height);
  ctx.lineTo(rect.x + r, rect.y + rect.height);
  ctx.quadraticCurveTo(rect.x, rect.y + rect.height, rect.x, rect.y + rect.height - r);
  ctx.lineTo(rect.x, rect.y + r);
  ctx.quadraticCurveTo(rect.x, rect.y, rect.x + r, rect.y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.shadowColor = 'transparent';
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.restore();
}

function drawLabel(ctx, text, x, y, {
  fontSize = 18,
  weight = 600,
  color = DIAGRAM_TEXT_DARK,
  align = 'center',
  baseline = 'middle',
  lineHeight = null,
} = {}) {
  const lines = String(text).split('\n');
  const lh = lineHeight || fontSize * 1.2;
  ctx.save();
  ctx.font = `${weight} ${fontSize}px ${DIAGRAM_FONT_FAMILY}`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  const startY = y - ((lines.length - 1) * lh) / 2;
  lines.forEach((line, index) => {
    ctx.fillText(line, x, startY + index * lh);
  });
  ctx.restore();
}

function drawArrowHead2D(ctx, from, to, color = DIAGRAM_LINE_COLOR, size = 11) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  ctx.save();
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.moveTo(to.x, to.y);
  ctx.lineTo(to.x - size * Math.cos(angle - Math.PI / 7), to.y - size * Math.sin(angle - Math.PI / 7));
  ctx.lineTo(to.x - size * Math.cos(angle + Math.PI / 7), to.y - size * Math.sin(angle + Math.PI / 7));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function strokePolyline(ctx, points, {
  color = DIAGRAM_LINE_COLOR,
  width = 3,
  arrow = false,
  dash = [],
} = {}) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.shadowColor = color;
  ctx.shadowBlur = dash.length ? 0 : 8;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
  ctx.restore();
  if (arrow) {
    drawArrowHead2D(ctx, points[points.length - 2], points[points.length - 1], color, width * 3.2);
  }
}

function pointAlongPolyline(points, t) {
  if (!points || points.length < 2) return null;
  const segments = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    const dx = points[i + 1].x - points[i].x;
    const dy = points[i + 1].y - points[i].y;
    const len = Math.hypot(dx, dy);
    segments.push({ start: points[i], end: points[i + 1], len, total });
    total += len;
  }
  if (total <= 0) return points[0];
  const target = total * Math.max(0, Math.min(1, t));
  const seg = segments.find((item) => target <= item.total + item.len) || segments[segments.length - 1];
  const local = seg.len > 0 ? (target - seg.total) / seg.len : 0;
  return {
    x: seg.start.x + (seg.end.x - seg.start.x) * local,
    y: seg.start.y + (seg.end.y - seg.start.y) * local,
  };
}

function buildDiagramLayout(width, height) {
  const view = { width: 1280, height: 980 };
  const scale = Math.min((width - 90) / view.width, (height - 70) / view.height);
  const offsetX = (width - view.width * scale) / 2;
  const offsetY = (height - view.height * scale) / 2;
  const mapRect = (x, y, w, h) => createRect(
    offsetX + x * scale,
    offsetY + y * scale,
    w * scale,
    h * scale,
  );
  const pt = (x, y) => ({ x: offsetX + x * scale, y: offsetY + y * scale });

  const boxes = {
    inputEmbedding: mapRect(246, 846, 170, 74),
    encoder: mapRect(228, 562, 220, 120),
    encoderAddNormBottom: mapRect(236, 482, 206, 56),
    encoderFeedForward: mapRect(244, 364, 190, 106),
    encoderAddNormTop: mapRect(236, 292, 206, 56),
    outputEmbedding: mapRect(724, 846, 176, 74),
    questionSetAttention: mapRect(698, 624, 246, 126),
    decoderAddNormSelf: mapRect(716, 546, 212, 56),
    decoderCrossAttention: mapRect(698, 426, 246, 108),
    decoderAddNormCross: mapRect(716, 354, 212, 56),
    decoder: mapRect(722, 238, 198, 104),
    decoderAddNormTop: mapRect(716, 164, 212, 56),
    maskHead: mapRect(604, 18, 102, 46),
    valueHead: mapRect(716, 18, 102, 46),
    scoreHead: mapRect(828, 18, 102, 46),
    countHead: mapRect(644, 80, 112, 46),
    slotSelectHead: mapRect(770, 80, 136, 46),
    outputBlock: mapRect(958, 44, 182, 64),
  };

  return {
    width,
    height,
    scale,
    boxes,
    frameEncoder: mapRect(180, 272, 310, 444),
    frameDecoder: mapRect(652, 148, 338, 618),
    frameHeads: mapRect(604, 6, 304, 132),
    inputSource: mapRect(58, 826, 152, 64),
    outputSourceA: mapRect(930, 832, 118, 58),
    outputSourceB: mapRect(1062, 818, 126, 72),
    inputPlus: pt(242, 772),
    outputPlus: pt(720, 772),
    inputWave: pt(188, 772),
    outputWave: pt(778, 772),
    inputArrowStart: pt(330, 946),
    outputArrowStart: pt(812, 946),
    outputProb: pt(776, 6),
    outputGroupLabel: pt(776, 6),
    inputCaption: pt(326, 964),
    outputCaption: pt(814, 964),
    leftNx: pt(128, 524),
    rightNx: pt(1036, 388),
    headHub: pt(822, 146),
    routes: {
      trip_to_input: [pt(206, 858), rectLeft(boxes.inputEmbedding, 0.52), rectCenter(boxes.inputEmbedding)],
      input_to_encoder: [rectTop(boxes.inputEmbedding), pt(312, 772), rectBottom(boxes.encoder)],
      encoder_to_output: [rectTop(boxes.encoderAddNormTop), pt(514, 292), pt(514, 426), rectLeft(boxes.decoderCrossAttention, 0.54)],
      output_to_qset: [rectTop(boxes.outputEmbedding), pt(794, 772), rectBottom(boxes.questionSetAttention)],
      qset_to_decoder: [rectTop(boxes.questionSetAttention), rectBottom(boxes.decoderCrossAttention)],
      decoder_to_mask: [rectTop(boxes.decoderAddNormTop), pt(822, 146), rectBottom(boxes.maskHead)],
      decoder_to_value: [rectTop(boxes.decoderAddNormTop), pt(822, 146), rectBottom(boxes.valueHead)],
      decoder_to_score: [rectTop(boxes.decoderAddNormTop), pt(822, 146), rectBottom(boxes.scoreHead)],
      decoder_to_count: [rectTop(boxes.decoderAddNormTop), pt(822, 146), rectBottom(boxes.countHead)],
      decoder_to_slot_select: [rectTop(boxes.decoderAddNormTop), pt(822, 146), rectBottom(boxes.slotSelectHead)],
      heads_to_output: [pt(890, 70), rectLeft(boxes.outputBlock, 0.5), rectCenter(boxes.outputBlock)],
    },
  };
}

function drawPlusCircle(ctx, center, radius, scale) {
  ctx.save();
  ctx.lineWidth = 2.5 * scale;
  ctx.strokeStyle = DIAGRAM_LINE_COLOR;
  ctx.shadowColor = DIAGRAM_LINE_COLOR;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(center.x - radius * 0.45, center.y);
  ctx.lineTo(center.x + radius * 0.45, center.y);
  ctx.moveTo(center.x, center.y - radius * 0.45);
  ctx.lineTo(center.x, center.y + radius * 0.45);
  ctx.stroke();
  ctx.restore();
}

function drawWaveCircle(ctx, center, radius, scale) {
  ctx.save();
  ctx.lineWidth = 2.5 * scale;
  ctx.strokeStyle = DIAGRAM_LINE_COLOR;
  ctx.shadowColor = DIAGRAM_LINE_COLOR;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 24; i += 1) {
    const t = i / 24;
    const px = center.x - radius * 0.62 + t * radius * 1.24;
    const py = center.y + Math.sin(t * Math.PI * 2) * radius * 0.32;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
}

function drawResidualLoop(ctx, points, scale, color = DIAGRAM_LINE_COLOR) {
  strokePolyline(ctx, points, { color, width: 2.6 * scale, arrow: true });
}

function drawDiagramBox(ctx, rect, label, options, isActive, isHovered) {
  const fill = isActive
    ? lerpColorHex(options.fill, '#fef3c7', 0.24)
    : (isHovered ? lerpColorHex(options.fill, '#ffffff', 0.2) : options.fill);
  const border = isActive ? '#0f172a' : '#111827';
  drawRoundedRect(ctx, rect, 14 * options.scale, fill, border, 2.8 * options.scale, isActive);
  if (isActive) {
    ctx.save();
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.36)';
    ctx.lineWidth = 7 * options.scale;
    ctx.strokeRect(rect.x - 4 * options.scale, rect.y - 4 * options.scale, rect.width + 8 * options.scale, rect.height + 8 * options.scale);
    ctx.restore();
  }
  drawLabel(ctx, label, rect.x + rect.width / 2, rect.y + rect.height / 2, {
    fontSize: options.fontSize * options.scale,
    weight: 500,
    color: '#171717',
    lineHeight: options.fontSize * options.scale * 1.15,
  });
}

function renderEncoderTabCanvas(ctx, width, height, selectedId) {
  const scale = Math.min(width / 320, height / 720);
  const centerX = width / 2;
  // 外框下沿下移，确保 concat 向上的连接线也被框线包住。
  const frame = createRect(centerX - 98 * scale, 92 * scale, 196 * scale, 384 * scale);
  const boxes = {
    encoderAddNormTop: createRect(centerX - 72 * scale, 120 * scale, 144 * scale, 36 * scale),
    encoderFeedForward: createRect(centerX - 84 * scale, 168 * scale, 168 * scale, 84 * scale),
    encoderAddNormBottom: createRect(centerX - 72 * scale, 268 * scale, 144 * scale, 36 * scale),
    encoder: createRect(centerX - 88 * scale, 332 * scale, 176 * scale, 86 * scale),
    concatPanel: createRect(centerX - 68 * scale, 480 * scale, 136 * scale, 38 * scale),
    inputEmbedding: createRect(centerX - 78 * scale, 528 * scale, 156 * scale, 62 * scale),
  };
  const labelXs = [
    centerX - 126 * scale,
    centerX - 42 * scale,
    centerX + 42 * scale,
    centerX + 126 * scale,
  ];
  const baselineY = boxes.inputEmbedding.y + boxes.inputEmbedding.height + 30 * scale;
  const featureY = baselineY + 18 * scale;
  const featureBoxes = [
    createRect(labelXs[0] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[1] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[2] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[3] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
  ];
  const featureDefs = [
    { id: 'x_rp', label: 'X_rp', fill: DIAGRAM_MODULE_COLORS.feedForward },
    { id: 'x_spCtx', label: 'X_hist', fill: DIAGRAM_MODULE_COLORS.attention },
    { id: 'x_env', label: 'X_env', fill: DIAGRAM_MODULE_COLORS.feedForward },
    { id: 'x_cand', label: 'X_cand', fill: DIAGRAM_MODULE_COLORS.attention },
  ];
  const rpTag = createRect(centerX - 120 * scale, featureY + 30 * scale, 108 * scale, 34 * scale);
  const spTag = createRect(centerX + 12 * scale, featureY + 30 * scale, 108 * scale, 34 * scale);
  const boxDefs = [
    ['encoderAddNormTop', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 16, scale }],
    ['encoderFeedForward', 'Feed\nForward', { fill: DIAGRAM_MODULE_COLORS.feedForward, fontSize: 20, scale }],
    ['encoderAddNormBottom', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 16, scale }],
    ['encoder', 'Multi-Head\nAttention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 18, scale }],
    ['concatPanel', 'Concat', { fill: DIAGRAM_MODULE_COLORS.neutral, fontSize: 18, scale }],
    ['inputEmbedding', 'Input\nEmbedding', { fill: DIAGRAM_MODULE_COLORS.embedding, fontSize: 18, scale }],
  ];

  diagramState.hitRegions = [];

  drawRoundedRect(ctx, frame, 22 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 3.2 * scale);
  drawLabel(ctx, 'N×', frame.x - 40 * scale, frame.y + frame.height * 0.56, {
    fontSize: 30 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });

  boxDefs.forEach(([id, label, options]) => {
    const rectBox = boxes[id];
    const isActive = diagramState.activeIds.has(id) || selectedId === id;
    const isHovered = diagramState.hoverId === id;
    drawDiagramBox(ctx, rectBox, label, options, isActive, isHovered);
    diagramState.hitRegions.push({ id, rect: rectBox });
  });

  const inputTargets = [0.2, 0.4, 0.6, 0.8].map((ratio) => ({
    x: boxes.inputEmbedding.x + boxes.inputEmbedding.width * ratio,
    y: boxes.inputEmbedding.y + boxes.inputEmbedding.height,
  }));

  strokePolyline(ctx, [rectTop(boxes.inputEmbedding), rectBottom(boxes.concatPanel)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  const concatOut = rectTop(boxes.concatPanel);
  const forkHub = { x: concatOut.x, y: concatOut.y - 10 * scale };
  const mhaBottomTargets = [0.2, 0.5, 0.8].map((ratio) => ({
    x: boxes.encoder.x + boxes.encoder.width * ratio,
    y: boxes.encoder.y + boxes.encoder.height,
  }));
  strokePolyline(ctx, [concatOut, forkHub], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: false });
  strokePolyline(ctx, [forkHub, mhaBottomTargets[1]], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });
  strokePolyline(ctx, [forkHub, { x: mhaBottomTargets[0].x, y: forkHub.y }, mhaBottomTargets[0]], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });
  strokePolyline(ctx, [forkHub, { x: mhaBottomTargets[2].x, y: forkHub.y }, mhaBottomTargets[2]], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });

  const encoderAddNormHoutInput = rectBottomAt(boxes.encoderAddNormBottom, 0.66);
  const encoderArrowElbowY = encoderAddNormHoutInput.y + 8 * scale;
  strokePolyline(ctx, [
    rectTop(boxes.encoder),
    { x: rectTop(boxes.encoder).x, y: encoderArrowElbowY },
    { x: encoderAddNormHoutInput.x, y: encoderArrowElbowY },
    encoderAddNormHoutInput,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.encoderAddNormBottom), rectBottom(boxes.encoderFeedForward)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.encoderFeedForward), rectBottom(boxes.encoderAddNormTop)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  const lowerNormTop = rectTop(boxes.encoderAddNormBottom);
  const upperNormLeftMid = rectLeft(boxes.encoderAddNormTop, 0.5);
  const branchLaneX = frame.x + 14 * scale;
  strokePolyline(ctx, [
    lowerNormTop,
    { x: branchLaneX, y: lowerNormTop.y },
    { x: branchLaneX, y: upperNormLeftMid.y },
    upperNormLeftMid,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });

  const bypassStart = rectLeft(boxes.concatPanel, 0.5);
  const bypassLaneX = frame.x + 14 * scale;
  const bypassEnd = rectLeft(boxes.encoderAddNormBottom, 0.5);
  strokePolyline(ctx, [
    bypassStart,
    { x: bypassLaneX, y: bypassStart.y },
    { x: bypassLaneX, y: bypassEnd.y },
    bypassEnd,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });

  strokePolyline(ctx, [
    rectRight(boxes.encoderAddNormTop, 0.52),
    { x: width - 18 * scale, y: rectRight(boxes.encoderAddNormTop, 0.52).y },
  ], { color: DIAGRAM_LINE_COLOR, width: 2.6 * scale, arrow: true });

  const inputArrowBendBaseY = boxes.inputEmbedding.y + boxes.inputEmbedding.height + 10 * scale;
  featureDefs.forEach((feature, index) => {
    const rectBox = featureBoxes[index];
    const isActive = diagramState.hoverId === feature.id || selectedId === feature.id;
    const featureFill = isActive ? lerpColorHex(feature.fill, '#ffffff', 0.16) : feature.fill;
    drawPill(
      ctx,
      rectBox,
      feature.label,
      featureFill,
      '#111827',
      scale,
      14,
      '#111827',
    );
    const start = rectTop(rectBox);
    const inputArrowBendY = inputArrowBendBaseY + (index === 0 || index === 3 ? 7 : 1) * scale;
    strokePolyline(ctx, [
      { x: start.x, y: start.y },
      { x: start.x, y: inputArrowBendY },
      { x: inputTargets[index].x, y: inputArrowBendY },
      inputTargets[index],
    ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
    diagramState.hitRegions.push({ id: feature.id, rect: rectBox });
  });

  const rpTagActive = diagramState.hoverId === 'tripPaper' || dashboardState.focusLabel === 'Trip / RP 输入';
  const spTagActive = diagramState.hoverId === 'spPaper' || dashboardState.focusLabel === 'SP Survey';
  drawPill(
    ctx,
    rpTag,
    'RP part',
    rpTagActive ? lerpColorHex(DIAGRAM_MODULE_COLORS.tagBlue, '#ffffff', 0.12) : DIAGRAM_MODULE_COLORS.tagBlue,
    '#111827',
    scale,
    14,
    '#111827',
  );
  drawPill(
    ctx,
    spTag,
    'SP part',
    spTagActive ? lerpColorHex(DIAGRAM_MODULE_COLORS.tagLavender, '#ffffff', 0.12) : DIAGRAM_MODULE_COLORS.tagLavender,
    '#111827',
    scale,
    14,
    '#111827',
  );

  const rpStart = rectTop(rpTag);
  const spStart = rectTop(spTag);
  const rpArrowBendY = featureBoxes[0].y + featureBoxes[0].height + 8 * scale;
  const spArrowBendY = featureBoxes[0].y + featureBoxes[0].height + 16 * scale;
  const rpHubX = (featureBoxes[0].x + featureBoxes[0].width / 2 + featureBoxes[2].x + featureBoxes[2].width / 2) / 2 - 6 * scale;
  const spHubX = (featureBoxes[1].x + featureBoxes[1].width / 2 + featureBoxes[3].x + featureBoxes[3].width / 2) / 2 + 6 * scale;
  strokePolyline(ctx, [
    { x: rpStart.x, y: rpStart.y },
    { x: rpStart.x, y: rpArrowBendY },
    { x: rpHubX, y: rpArrowBendY },
  ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: false });
  [0, 2].forEach((idx) => {
    const toX = featureBoxes[idx].x + featureBoxes[idx].width / 2;
    const toY = featureBoxes[idx].y + featureBoxes[idx].height;
    strokePolyline(ctx, [
      { x: rpHubX, y: rpArrowBendY },
      { x: toX, y: rpArrowBendY },
      { x: toX, y: toY },
    ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: true });
  });
  strokePolyline(ctx, [
    { x: spStart.x, y: spStart.y },
    { x: spStart.x, y: spArrowBendY },
    { x: spHubX, y: spArrowBendY },
  ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: false });
  [1, 3].forEach((idx) => {
    const toX = featureBoxes[idx].x + featureBoxes[idx].width / 2;
    const toY = featureBoxes[idx].y + featureBoxes[idx].height;
    strokePolyline(ctx, [
      { x: spHubX, y: spArrowBendY },
      { x: toX, y: spArrowBendY },
      { x: toX, y: toY },
    ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: true });
  });

  diagramState.hitRegions.push({ id: 'tripPaper', rect: rpTag });
  diagramState.hitRegions.push({ id: 'spPaper', rect: spTag });
}

function renderDecoderTabCanvas(ctx, width, height, selectedId) {
  const scale = Math.min(width / 430, height / 980);
  const centerX = width / 2;
  const yShift = 64 * scale;
  const lowerDecoderShift = 24 * scale;
  const frameHeads = createRect(centerX - 138 * scale, 34 * scale + yShift, 276 * scale, 118 * scale);
  const frameDecoder = createRect(centerX - 114 * scale, 170 * scale + yShift, 228 * scale, 442 * scale + lowerDecoderShift);

  const boxes = {
    countHead: createRect(centerX - 132 * scale, 70 * scale + yShift, 80 * scale, 34 * scale),
    slotSelectHead: createRect(centerX - 40 * scale, 70 * scale + yShift, 80 * scale, 34 * scale),
    maskHead: createRect(centerX + 52 * scale, 70 * scale + yShift, 80 * scale, 34 * scale),
    valueHead: createRect(centerX - 86 * scale, 112 * scale + yShift, 80 * scale, 34 * scale),
    scoreHead: createRect(centerX + 6 * scale, 112 * scale + yShift, 80 * scale, 34 * scale),
    decoderAddNormTop: createRect(centerX - 74 * scale, 206 * scale + yShift, 148 * scale, 34 * scale),
    decoder: createRect(centerX - 84 * scale, 246 * scale + yShift, 168 * scale, 70 * scale),
    decoderAddNormCross: createRect(centerX - 74 * scale, 332 * scale + yShift, 148 * scale, 34 * scale),
    decoderCrossAttention: createRect(centerX - 90 * scale, 372 * scale + yShift, 180 * scale, 78 * scale),
    decoderAddNormSelf: createRect(centerX - 74 * scale, 466 * scale + yShift + lowerDecoderShift, 148 * scale, 34 * scale),
    questionSetAttention: createRect(centerX - 90 * scale, 508 * scale + yShift + lowerDecoderShift, 180 * scale, 84 * scale),
    outputEmbedding: createRect(
      centerX - 80 * scale,
      frameDecoder.y + frameDecoder.height + 6 * scale,
      160 * scale,
      28 * scale,
    ),
  };

  const questionStrip = createRect(centerX - 142 * scale, 836 * scale + yShift, 284 * scale, 42 * scale);
  const hEncLabel = createRect(frameDecoder.x - 212 * scale, boxes.decoderCrossAttention.y - 40 * scale, 120 * scale, 24 * scale);

  const labelXs = [
    centerX - 126 * scale,
    centerX - 42 * scale,
    centerX + 42 * scale,
    centerX + 126 * scale,
  ];
  const rpTag = createRect(centerX - 120 * scale, questionStrip.y - 56 * scale, 108 * scale, 34 * scale);
  const spTag = createRect(centerX + 12 * scale, questionStrip.y - 56 * scale, 108 * scale, 34 * scale);
  const featureY = rpTag.y - 68 * scale;
  const featureBoxes = [
    createRect(labelXs[0] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[1] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[2] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
    createRect(labelXs[3] - 38 * scale, featureY - 16 * scale, 76 * scale, 32 * scale),
  ];
  const featureDefs = [
    { id: 'x_rp', label: 'X_rp', fill: DIAGRAM_MODULE_COLORS.feedForward },
    { id: 'x_spCtx', label: 'X_hist', fill: DIAGRAM_MODULE_COLORS.attention },
    { id: 'x_env', label: 'X_env', fill: DIAGRAM_MODULE_COLORS.feedForward },
    { id: 'x_cand', label: 'X_cand', fill: DIAGRAM_MODULE_COLORS.attention },
  ];

  const boxDefs = [
    ['countHead', 'Count', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 12, scale }],
    ['slotSelectHead', 'Slot select', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 11, scale }],
    ['maskHead', 'Mask', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 12, scale }],
    ['valueHead', 'Value', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 12, scale }],
    ['scoreHead', 'Score', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 12, scale }],
    ['decoderAddNormTop', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 15, scale }],
    ['decoder', 'Feed\nForward', { fill: DIAGRAM_MODULE_COLORS.feedForward, fontSize: 18, scale }],
    ['decoderAddNormCross', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 15, scale }],
    ['decoderCrossAttention', 'Multi-Head\nAttention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 17, scale }],
    ['decoderAddNormSelf', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 15, scale }],
    ['questionSetAttention', 'Question-Set\nSelf-Attention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 16, scale }],
    ['outputEmbedding', 'Embedding&Concat', { fill: DIAGRAM_MODULE_COLORS.embedding, fontSize: 16, scale }],
  ];

  diagramState.hitRegions = [];

  drawRoundedRect(ctx, frameHeads, 20 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 2.8 * scale);
  drawRoundedRect(ctx, frameDecoder, 22 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 3.2 * scale);
  drawLabel(ctx, 'Head', frameHeads.x + 14 * scale, frameHeads.y + 18 * scale, {
    fontSize: 14 * scale,
    weight: 600,
    color: DIAGRAM_LINE_COLOR,
    align: 'left',
  });
  drawLabel(ctx, 'N×', frameDecoder.x + frameDecoder.width + 36 * scale, frameDecoder.y + frameDecoder.height * 0.52, {
    fontSize: 30 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });

  const questionActive = diagramState.hoverId === 'spPaper' || selectedId === 'spPaper' || dashboardState.focusId === 'spPaper';
  const questionFill = questionActive
    ? lerpColorHex(DIAGRAM_MODULE_COLORS.neutral, '#ffffff', 0.12)
    : DIAGRAM_MODULE_COLORS.neutral;
  drawRoundedRect(ctx, questionStrip, 4 * scale, questionFill, '#111827', 2.2 * scale);
  drawLabel(ctx, 'Question1, question2, ..., question n', questionStrip.x + questionStrip.width / 2, questionStrip.y + questionStrip.height / 2, {
    fontSize: 12 * scale,
    weight: 500,
    color: '#111827',
  });
  diagramState.hitRegions.push({ id: 'spPaper', rect: questionStrip });

  strokePolyline(ctx, [rectTop(boxes.questionSetAttention), rectBottom(boxes.decoderAddNormSelf)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.decoderAddNormSelf), rectBottom(boxes.decoderCrossAttention)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.decoderCrossAttention), rectBottom(boxes.decoderAddNormCross)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.decoderAddNormCross), rectBottom(boxes.decoder)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });
  strokePolyline(ctx, [rectTop(boxes.decoder), rectBottom(boxes.decoderAddNormTop)], { color: DIAGRAM_LINE_COLOR, width: 2.8 * scale, arrow: true });

  const headHub = { x: centerX, y: frameHeads.y + frameHeads.height };
  strokePolyline(ctx, [rectTop(boxes.decoderAddNormTop), headHub], { color: DIAGRAM_LINE_COLOR, width: 2.6 * scale, arrow: true });
  ['countHead', 'slotSelectHead', 'maskHead', 'valueHead', 'scoreHead'].forEach((headId) => {
    strokePolyline(ctx, [headHub, rectBottom(boxes[headId])], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  });

  drawLabel(ctx, 'encoder', hEncLabel.x + hEncLabel.width / 2, hEncLabel.y + hEncLabel.height / 2, {
    fontSize: 14 * scale,
    weight: 500,
    color: '#111827',
  });
  const encLaneX = frameDecoder.x - 22 * scale;
  const encStart = {
    x: hEncLabel.x + hEncLabel.width,
    y: hEncLabel.y + hEncLabel.height + 8 * scale,
  };
  const encBranchY = boxes.decoderCrossAttention.y + boxes.decoderCrossAttention.height + 14 * scale;
  const encTargetLeft = {
    x: boxes.decoderCrossAttention.x + boxes.decoderCrossAttention.width * 0.18,
    y: boxes.decoderCrossAttention.y + boxes.decoderCrossAttention.height,
  };
  const encTargetMid = {
    x: boxes.decoderCrossAttention.x + boxes.decoderCrossAttention.width * 0.40,
    y: boxes.decoderCrossAttention.y + boxes.decoderCrossAttention.height,
  };
  strokePolyline(ctx, [
    encStart,
    { x: encLaneX, y: encStart.y },
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  strokePolyline(ctx, [
    { x: encLaneX, y: encStart.y },
    { x: encLaneX, y: encBranchY },
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: false });
  strokePolyline(ctx, [
    { x: encLaneX, y: encBranchY },
    { x: encTargetMid.x, y: encBranchY },
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: false });
  strokePolyline(ctx, [
    { x: encTargetLeft.x, y: encBranchY },
    encTargetLeft,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  strokePolyline(ctx, [
    { x: encTargetMid.x, y: encBranchY },
    encTargetMid,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });

  drawResidualLoop(ctx, [
    { x: boxes.questionSetAttention.x + boxes.questionSetAttention.width * 0.84, y: rectRight(boxes.questionSetAttention, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.questionSetAttention, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.decoderAddNormSelf, 0.5).y },
    rectRight(boxes.decoderAddNormSelf, 0.5),
  ], scale);
  drawResidualLoop(ctx, [
    { x: boxes.decoderCrossAttention.x + boxes.decoderCrossAttention.width * 0.84, y: rectRight(boxes.decoderCrossAttention, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.decoderCrossAttention, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.decoderAddNormCross, 0.5).y },
    rectRight(boxes.decoderAddNormCross, 0.5),
  ], scale);
  drawResidualLoop(ctx, [
    { x: boxes.decoder.x + boxes.decoder.width * 0.84, y: rectRight(boxes.decoder, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.decoder, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 16 * scale, y: rectRight(boxes.decoderAddNormTop, 0.5).y },
    rectRight(boxes.decoderAddNormTop, 0.5),
  ], scale);

  boxDefs.forEach(([id, label, options]) => {
    const rectBox = boxes[id];
    const isActive = diagramState.activeIds.has(id) || selectedId === id;
    const isHovered = diagramState.hoverId === id;
    drawDiagramBox(ctx, rectBox, label, options, isActive, isHovered);
    diagramState.hitRegions.push({ id, rect: rectBox });
  });

  const mergeY = featureBoxes[0].y - 10 * scale;
  const mergeLeftX = featureBoxes[0].x + featureBoxes[0].width / 2;
  const mergeRightX = featureBoxes[3].x + featureBoxes[3].width / 2;
  const mergeHub = { x: centerX, y: mergeY };
  const maskedBottom = rectBottom(boxes.questionSetAttention);
  const embeddingBottom = rectBottom(boxes.outputEmbedding);
  const embeddingTop = rectTop(boxes.outputEmbedding);
  featureDefs.forEach((feature, index) => {
    const rectBox = featureBoxes[index];
    const isActive = diagramState.hoverId === feature.id || selectedId === feature.id;
    const featureFill = isActive ? lerpColorHex(feature.fill, '#ffffff', 0.14) : feature.fill;
    drawPill(ctx, rectBox, feature.label, featureFill, '#111827', scale, 14, '#111827');
    const start = rectTop(rectBox);
    strokePolyline(ctx, [
      { x: start.x, y: start.y },
      { x: start.x, y: mergeY },
    ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: false });
    diagramState.hitRegions.push({ id: feature.id, rect: rectBox });
  });
  strokePolyline(ctx, [
    { x: mergeLeftX, y: mergeY },
    { x: mergeRightX, y: mergeY },
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: false });
  strokePolyline(ctx, [
    mergeHub,
    embeddingBottom,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });
  strokePolyline(ctx, [
    embeddingTop,
    maskedBottom,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });

  const rpTagActive = diagramState.hoverId === 'tripPaper' || dashboardState.focusLabel === 'Trip / RP 输入';
  const spTagActive = diagramState.hoverId === 'spPaper' || dashboardState.focusLabel === 'SP Survey';
  drawPill(
    ctx,
    rpTag,
    'RP part',
    rpTagActive ? lerpColorHex(DIAGRAM_MODULE_COLORS.tagBlue, '#ffffff', 0.12) : DIAGRAM_MODULE_COLORS.tagBlue,
    '#111827',
    scale,
    14,
    '#111827',
  );
  drawPill(
    ctx,
    spTag,
    'SP part',
    spTagActive ? lerpColorHex(DIAGRAM_MODULE_COLORS.tagLavender, '#ffffff', 0.12) : DIAGRAM_MODULE_COLORS.tagLavender,
    '#111827',
    scale,
    14,
    '#111827',
  );
  diagramState.hitRegions.push({ id: 'tripPaper', rect: rpTag });
  diagramState.hitRegions.push({ id: 'spPaper', rect: spTag });

  const rpStart = rectTop(rpTag);
  const spStart = rectTop(spTag);
  const xBottomY = featureBoxes[0].y + featureBoxes[0].height;
  const partArrowBendY = (rpStart.y + xBottomY) / 2;
  [0, 1, 2, 3].forEach((idx) => {
    const toX = featureBoxes[idx].x + featureBoxes[idx].width / 2;
    const toY = featureBoxes[idx].y + featureBoxes[idx].height;
    strokePolyline(ctx, [
      { x: rpStart.x, y: rpStart.y },
      { x: rpStart.x, y: partArrowBendY },
      { x: toX, y: partArrowBendY },
      { x: toX, y: toY },
    ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: true });
  });
  [0, 1, 2, 3].forEach((idx) => {
    const toX = featureBoxes[idx].x + featureBoxes[idx].width / 2;
    const toY = featureBoxes[idx].y + featureBoxes[idx].height;
    strokePolyline(ctx, [
      { x: spStart.x, y: spStart.y },
      { x: spStart.x, y: partArrowBendY },
      { x: toX, y: partArrowBendY },
      { x: toX, y: toY },
    ], { color: DIAGRAM_LINE_COLOR, width: 2 * scale, arrow: true });
  });

  const spCenter = rectBottom(spTag);
  strokePolyline(ctx, [
    { x: spCenter.x, y: questionStrip.y },
    spCenter,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });

  const leftLaneX = frameDecoder.x - 116 * scale;
  const rightLaneValueX = frameDecoder.x + frameDecoder.width + 100 * scale;
  const rightLaneCountX = frameDecoder.x + frameDecoder.width + 132 * scale;
  const rightLaneSlotX = frameDecoder.x + frameDecoder.width + 116 * scale;
  const qLeft = rectLeft(questionStrip, 0.56);
  const qRightUpper = rectRight(questionStrip, 0.40);
  const qRightMiddle = rectRight(questionStrip, 0.56);
  const qRightLower = rectRight(questionStrip, 0.74);
  strokePolyline(ctx, [
    rectLeft(boxes.maskHead, 0.5),
    { x: leftLaneX, y: rectLeft(boxes.maskHead, 0.5).y },
    { x: leftLaneX, y: qLeft.y },
    qLeft,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  strokePolyline(ctx, [
    rectRight(boxes.valueHead, 0.5),
    { x: rightLaneValueX, y: rectRight(boxes.valueHead, 0.5).y },
    { x: rightLaneValueX, y: qRightUpper.y },
    qRightUpper,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  strokePolyline(ctx, [
    rectRight(boxes.slotSelectHead, 0.5),
    { x: rightLaneSlotX, y: rectRight(boxes.slotSelectHead, 0.5).y },
    { x: rightLaneSlotX, y: qRightMiddle.y },
    qRightMiddle,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
  strokePolyline(ctx, [
    rectRight(boxes.countHead, 0.5),
    { x: rightLaneCountX, y: rectRight(boxes.countHead, 0.5).y },
    { x: rightLaneCountX, y: qRightLower.y },
    qRightLower,
  ], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
}

function renderDiagramCanvas() {
  if (!diagramCanvas || !diagramCtx) return;
  const rect = diagramCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.round(rect.width * dpr);
  const targetHeight = Math.round(rect.height * dpr);
  if (diagramCanvas.width !== targetWidth || diagramCanvas.height !== targetHeight) {
    diagramCanvas.width = targetWidth;
    diagramCanvas.height = targetHeight;
  }

  diagramCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  diagramCtx.clearRect(0, 0, rect.width, rect.height);
  const selectedId = diagramState.hoverId || diagramState.selectedId;
  if (demoFlowState.mode === 'idle') {
    if (diagramViewMode === 'decoder') renderDecoderTabCanvas(diagramCtx, rect.width, rect.height, selectedId);
    else renderEncoderTabCanvas(diagramCtx, rect.width, rect.height, selectedId);
    return;
  }

  const layout = buildDiagramLayout(rect.width, rect.height);
  const { boxes, frameEncoder, frameDecoder, frameHeads, routes } = layout;
  const scale = layout.scale;
  diagramState.hitRegions = [];

  drawRoundedRect(diagramCtx, frameEncoder, 26 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 4 * scale);
  drawRoundedRect(diagramCtx, frameDecoder, 26 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 4 * scale);
  drawRoundedRect(diagramCtx, frameHeads, 20 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 3 * scale);

  drawLabel(diagramCtx, 'Multi-Head\nOutputs', layout.outputGroupLabel.x, layout.outputGroupLabel.y + 10 * scale, {
    fontSize: 22 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 26 * scale,
  });
  drawLabel(diagramCtx, 'Inputs', layout.inputCaption.x, layout.inputCaption.y, {
    fontSize: 24 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });
  drawLabel(diagramCtx, 'Question Block\n(set output)', layout.outputCaption.x, layout.outputCaption.y, {
    fontSize: 23 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 28 * scale,
  });
  drawLabel(diagramCtx, 'N×', layout.leftNx.x, layout.leftNx.y, {
    fontSize: 30 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });
  drawLabel(diagramCtx, 'N×', layout.rightNx.x, layout.rightNx.y, {
    fontSize: 30 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });
  drawLabel(diagramCtx, 'Positional\nEncoding', layout.inputWave.x - 48 * scale, layout.inputWave.y + 10 * scale, {
    fontSize: 18 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 21 * scale,
  });
  drawLabel(diagramCtx, 'Positional\nEncoding', layout.outputWave.x + 92 * scale, layout.outputWave.y + 10 * scale, {
    fontSize: 18 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 21 * scale,
  });

  strokePolyline(diagramCtx, [layout.inputArrowStart, rectBottom(boxes.inputEmbedding)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [layout.outputArrowStart, rectBottom(boxes.outputEmbedding)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  drawWaveCircle(diagramCtx, layout.inputWave, 16 * scale, scale);
  drawWaveCircle(diagramCtx, layout.outputWave, 16 * scale, scale);
  drawPlusCircle(diagramCtx, layout.inputPlus, 16 * scale, scale);
  drawPlusCircle(diagramCtx, layout.outputPlus, 16 * scale, scale);
  strokePolyline(diagramCtx, [rectTop(boxes.inputEmbedding), layout.inputPlus, rectBottom(boxes.encoder)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.outputEmbedding), layout.outputPlus, rectBottom(boxes.questionSetAttention)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });

  const encoderAddNormHoutInput = rectBottomAt(boxes.encoderAddNormBottom, 0.66);
  const encoderArrowElbowY = encoderAddNormHoutInput.y + 8 * scale;
  strokePolyline(diagramCtx, [
    rectTop(boxes.encoder),
    { x: rectTop(boxes.encoder).x, y: encoderArrowElbowY },
    { x: encoderAddNormHoutInput.x, y: encoderArrowElbowY },
    encoderAddNormHoutInput,
  ], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.encoderAddNormBottom), rectBottom(boxes.encoderFeedForward)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.encoderFeedForward), rectBottom(boxes.encoderAddNormTop)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });

  strokePolyline(diagramCtx, [rectTop(boxes.questionSetAttention), rectBottom(boxes.decoderAddNormSelf)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.decoderAddNormSelf), rectBottom(boxes.decoderCrossAttention)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.decoderCrossAttention), rectBottom(boxes.decoderAddNormCross)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.decoderAddNormCross), rectBottom(boxes.decoder)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.decoder), rectBottom(boxes.decoderAddNormTop)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectTop(boxes.decoderAddNormTop), layout.headHub], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: false });
  strokePolyline(diagramCtx, [layout.headHub, rectBottom(boxes.maskHead)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [layout.headHub, rectBottom(boxes.valueHead)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [layout.headHub, rectBottom(boxes.scoreHead)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [layout.headHub, rectBottom(boxes.countHead)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [layout.headHub, rectBottom(boxes.slotSelectHead)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectRight(boxes.valueHead, 0.55), rectLeft(boxes.outputBlock, 0.4)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectRight(boxes.slotSelectHead, 0.55), rectLeft(boxes.outputBlock, 0.55)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  strokePolyline(diagramCtx, [rectRight(boxes.countHead, 0.5), rectLeft(boxes.outputBlock, 0.7)], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: true });
  const encoderRouteStart = rectRight(boxes.encoderAddNormTop, 0.42);
  const encoderRouteElbow = { x: frameEncoder.x + frameEncoder.width + 46 * scale, y: encoderRouteStart.y };
  const encoderRouteDrop = { x: frameEncoder.x + frameEncoder.width + 46 * scale, y: rectLeft(boxes.decoderCrossAttention, 0.5).y };
  const encoderRouteEnd = rectLeft(boxes.decoderCrossAttention, 0.5);
  strokePolyline(diagramCtx, [encoderRouteStart, encoderRouteElbow, encoderRouteDrop, encoderRouteEnd], { color: DIAGRAM_LINE_COLOR, width: 3 * scale, arrow: false });
  drawLabel(diagramCtx, 'encoder', (encoderRouteStart.x + encoderRouteElbow.x) / 2, encoderRouteStart.y - 14 * scale, {
    fontSize: 16 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });

  drawResidualLoop(diagramCtx, [
    { x: boxes.encoder.x + boxes.encoder.width * 0.16, y: rectLeft(boxes.encoder, 0.5).y },
    { x: frameEncoder.x + 18 * scale, y: rectLeft(boxes.encoder, 0.5).y },
    { x: frameEncoder.x + 18 * scale, y: rectLeft(boxes.encoderAddNormBottom, 0.5).y },
    rectLeft(boxes.encoderAddNormBottom, 0.5),
  ], scale);
  drawResidualLoop(diagramCtx, [
    { x: boxes.encoderFeedForward.x + boxes.encoderFeedForward.width * 0.18, y: rectLeft(boxes.encoderFeedForward, 0.5).y },
    { x: frameEncoder.x + 18 * scale, y: rectLeft(boxes.encoderFeedForward, 0.5).y },
    { x: frameEncoder.x + 18 * scale, y: rectLeft(boxes.encoderAddNormTop, 0.5).y },
    rectLeft(boxes.encoderAddNormTop, 0.5),
  ], scale);
  drawResidualLoop(diagramCtx, [
    rectRight(boxes.decoderAddNormSelf, 0.5),
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.decoderAddNormSelf, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.questionSetAttention, 0.5).y },
    { x: boxes.questionSetAttention.x + boxes.questionSetAttention.width * 0.84, y: rectRight(boxes.questionSetAttention, 0.5).y },
  ], scale);
  drawResidualLoop(diagramCtx, [
    rectRight(boxes.decoderAddNormCross, 0.5),
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.decoderAddNormCross, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.decoderCrossAttention, 0.5).y },
    { x: boxes.decoderCrossAttention.x + boxes.decoderCrossAttention.width * 0.84, y: rectRight(boxes.decoderCrossAttention, 0.5).y },
  ], scale);
  drawResidualLoop(diagramCtx, [
    rectRight(boxes.decoderAddNormTop, 0.5),
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.decoderAddNormTop, 0.5).y },
    { x: frameDecoder.x + frameDecoder.width - 18 * scale, y: rectRight(boxes.decoder, 0.5).y },
    { x: boxes.decoder.x + boxes.decoder.width * 0.84, y: rectRight(boxes.decoder, 0.5).y },
  ], scale);

  const routeColors = {
    trip_to_input: '#61dafb',
    input_to_encoder: '#61dafb',
    encoder_to_output: '#8b9df7',
    output_to_qset: '#61dafb',
    qset_to_decoder: '#fbbf24',
    decoder_to_mask: '#f97316',
    decoder_to_value: '#14b8a6',
    decoder_to_score: '#a855f7',
    decoder_to_count: '#ef4444',
    decoder_to_slot_select: '#fbbf24',
    heads_to_output: '#7ee787',
  };
  Object.entries(routes).forEach(([routeId, points]) => {
    const isActiveRoute = diagramState.activeConnectorId === routeId;
    strokePolyline(diagramCtx, points, {
      color: isActiveRoute ? DIAGRAM_LINE_COLOR : DIAGRAM_LINE_FAINT,
      width: isActiveRoute ? 4.8 * scale : 2.1 * scale,
      arrow: false,
      dash: isActiveRoute ? [] : [6 * scale, 7 * scale],
    });
    if (isActiveRoute && diagramState.routeDuration > 0) {
      const progress = Math.min((performance.now() - diagramState.routeStarted) / diagramState.routeDuration, 1);
      const point = pointAlongPolyline(points, progress);
      if (point) {
        diagramCtx.save();
        diagramCtx.fillStyle = routeColors[routeId];
        diagramCtx.shadowColor = routeColors[routeId];
        diagramCtx.shadowBlur = 18 * scale;
        diagramCtx.beginPath();
        diagramCtx.arc(point.x, point.y, 7 * scale, 0, Math.PI * 2);
        diagramCtx.fill();
        diagramCtx.restore();
      }
    }
  });

  const boxDefs = [
    ['maskHead', 'Mask\nHead', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 16, scale }],
    ['valueHead', 'Value\nHead', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 16, scale }],
    ['scoreHead', 'Score\nHead', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 16, scale }],
    ['countHead', 'Count\nHead', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 16, scale }],
    ['slotSelectHead', 'Slot Select\nHead', { fill: DIAGRAM_MODULE_COLORS.head, fontSize: 13, scale }],
    ['outputBlock', '生成题组\nQuestion Block', { fill: DIAGRAM_MODULE_COLORS.neutral, fontSize: 15, scale }],
    ['decoderAddNormTop', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 17, scale }],
    ['decoder', 'Feed\nForward', { fill: DIAGRAM_MODULE_COLORS.feedForward, fontSize: 20, scale }],
    ['decoderAddNormCross', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 17, scale }],
    ['decoderCrossAttention', 'Multi-Head\nAttention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 18, scale }],
    ['decoderAddNormSelf', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 17, scale }],
    ['questionSetAttention', 'Question-Set\nSelf-Attention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 17, scale }],
    ['encoderAddNormTop', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 17, scale }],
    ['encoderFeedForward', 'Feed\nForward', { fill: DIAGRAM_MODULE_COLORS.feedForward, fontSize: 20, scale }],
    ['encoderAddNormBottom', 'Add & Norm', { fill: DIAGRAM_MODULE_COLORS.addNorm, fontSize: 17, scale }],
    ['encoder', 'Multi-Head\nAttention', { fill: DIAGRAM_MODULE_COLORS.attention, fontSize: 18, scale }],
    ['inputEmbedding', 'Input\nEmbedding', { fill: DIAGRAM_MODULE_COLORS.embedding, fontSize: 18, scale }],
    ['outputEmbedding', 'Embedding&\nConcat', { fill: DIAGRAM_MODULE_COLORS.embedding, fontSize: 18, scale }],
  ];

  boxDefs.forEach(([id, label, options]) => {
    const rectBox = boxes[id];
    const isActive = diagramState.activeIds.has(id) || selectedId === id;
    const isHovered = diagramState.hoverId === id;
    drawDiagramBox(diagramCtx, rectBox, label, options, isActive, isHovered);
    diagramState.hitRegions.push({ id, rect: rectBox });
  });

  drawRoundedRect(diagramCtx, layout.inputSource, 4 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 2.3 * scale);
  drawRoundedRect(diagramCtx, layout.outputSourceA, 4 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 2.3 * scale);
  drawRoundedRect(diagramCtx, layout.outputSourceB, 4 * scale, 'rgba(255,255,255,0)', DIAGRAM_LINE_COLOR, 2.3 * scale);
  drawLabel(diagramCtx, 'RP/SP', layout.inputSource.x + layout.inputSource.width / 2, layout.inputSource.y + layout.inputSource.height / 2, {
    fontSize: 21 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
  });
  drawLabel(diagramCtx, 'SP\nquestion1', layout.outputSourceA.x + layout.outputSourceA.width / 2, layout.outputSourceA.y + layout.outputSourceA.height / 2, {
    fontSize: 15 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 18 * scale,
  });
  drawLabel(diagramCtx, 'SP\nquestion2...', layout.outputSourceB.x + layout.outputSourceB.width / 2, layout.outputSourceB.y + layout.outputSourceB.height / 2, {
    fontSize: 15 * scale,
    weight: 500,
    color: DIAGRAM_LINE_COLOR,
    lineHeight: 18 * scale,
  });

  drawDiagramOverlay(diagramCtx, layout, scale);
}

function findDiagramHit(clientX, clientY) {
  if (!diagramCanvas) return null;
  const rect = diagramCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  return diagramState.hitRegions.find((item) => (
    x >= item.rect.x
    && x <= item.rect.x + item.rect.width
    && y >= item.rect.y
    && y <= item.rect.y + item.rect.height
  )) || null;
}

function setDiagramSelection(id = '') {
  diagramState.selectedId = CANVAS_MODULE_IDS.has(id) ? id : '';
}

const demoFlowState = {
  mode: 'idle',
  step: 'idle',
  started: 0,
  duration: 0,
  headFocus: 'maskHead',
  countDecision: 'pending',
  sampleCount: 1,
};

let diagramViewMode = 'encoder';

function syncDiagramTabs() {
  if (encoderTabBtn) encoderTabBtn.classList.toggle('is-active', diagramViewMode === 'encoder');
  if (decoderTabBtn) decoderTabBtn.classList.toggle('is-active', diagramViewMode === 'decoder');
}

function setDiagramViewMode(mode = 'encoder', { force = false } = {}) {
  if (!force && diagramViewMode === mode) return;
  diagramViewMode = mode === 'decoder' ? 'decoder' : 'encoder';
  syncDiagramTabs();
  renderDiagramCanvas();
}

function setDiagramOverlay(mode = 'idle', step = 'idle', extras = {}) {
  demoFlowState.mode = mode;
  demoFlowState.step = step;
  demoFlowState.started = performance.now();
  demoFlowState.duration = extras.duration || 1000;
  demoFlowState.headFocus = extras.headFocus || demoFlowState.headFocus;
  demoFlowState.countDecision = extras.countDecision || demoFlowState.countDecision;
  demoFlowState.sampleCount = extras.sampleCount || demoFlowState.sampleCount;
}

function clearDiagramOverlay() {
  demoFlowState.mode = 'idle';
  demoFlowState.step = 'idle';
  demoFlowState.duration = 0;
}

function drawPill(ctx, rect, label, fill, stroke, scale, fontSize = 16, textColor = DIAGRAM_TEXT_DARK) {
  drawRoundedRect(ctx, rect, 12 * scale, fill, stroke, 2.2 * scale);
  drawLabel(ctx, label, rect.x + rect.width / 2, rect.y + rect.height / 2, {
    fontSize: fontSize * scale,
    weight: 600,
    color: textColor,
    lineHeight: fontSize * scale * 1.15,
  });
}

function drawCircleNode(ctx, x, y, radius, fill, stroke) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1.6, radius * 0.16);
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.restore();
}

function drawNeuralPanel(ctx, rect, scale, title, accent = '#60a5fa') {
  drawRoundedRect(ctx, rect, 18 * scale, 'rgba(255,255,255,0.92)', accent, 2.6 * scale, true);
  drawLabel(ctx, title, rect.x + rect.width / 2, rect.y + 18 * scale, {
    fontSize: 14 * scale,
    weight: 700,
    color: '#0f172a',
  });
  const layers = [3, 5, 4];
  const xStep = rect.width / (layers.length + 1);
  const nodes = [];
  layers.forEach((count, layerIdx) => {
    const x = rect.x + xStep * (layerIdx + 1);
    const yStep = rect.height / (count + 1);
    const group = [];
    for (let i = 0; i < count; i += 1) {
      group.push({ x, y: rect.y + 38 * scale + yStep * (i + 1) * 0.82 });
    }
    nodes.push(group);
  });
  ctx.save();
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.28)';
  ctx.lineWidth = 1.3 * scale;
  for (let l = 0; l < nodes.length - 1; l += 1) {
    nodes[l].forEach((from) => {
      nodes[l + 1].forEach((to) => {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });
    });
  }
  ctx.restore();
  nodes.flat().forEach((node) => drawCircleNode(ctx, node.x, node.y, 7 * scale, '#f8fafc', accent));
}

function drawInfoCard(ctx, rect, title, lines, scale, accent = '#64748b') {
  drawRoundedRect(ctx, rect, 16 * scale, 'rgba(255,255,255,0.9)', accent, 2.2 * scale);
  drawLabel(ctx, title, rect.x + rect.width / 2, rect.y + 18 * scale, {
    fontSize: 14 * scale,
    weight: 700,
    color: '#0f172a',
  });
  drawLabel(ctx, lines.join('\n'), rect.x + rect.width / 2, rect.y + rect.height / 2 + 8 * scale, {
    fontSize: 13 * scale,
    weight: 500,
    color: '#1e293b',
    lineHeight: 17 * scale,
  });
}

function drawDiagramOverlay(ctx, layout, scale) {
  if (demoFlowState.mode === 'idle') return;
  const progress = demoFlowState.duration > 0
    ? Math.min((performance.now() - demoFlowState.started) / demoFlowState.duration, 1)
    : 1;
  const boxes = layout.boxes;

  if (demoFlowState.mode === 'dataFlow') {
    if (demoFlowState.step === 'rawInputs') {
      const chips = [
        { rect: createRect(layout.inputSource.x + 10 * scale, layout.inputSource.y - 86 * scale, 120 * scale, 34 * scale), label: 'X_rp', fill: '#dbeafe' },
        { rect: createRect(layout.inputSource.x + 140 * scale, layout.inputSource.y - 86 * scale, 126 * scale, 34 * scale), label: 'X_env', fill: '#dcfce7' },
        { rect: createRect(layout.outputSourceA.x - 18 * scale, layout.outputSourceA.y - 86 * scale, 146 * scale, 34 * scale), label: 'X_cand', fill: '#fde68a' },
        { rect: createRect(layout.outputSourceB.x - 6 * scale, layout.outputSourceB.y - 86 * scale, 146 * scale, 34 * scale), label: 'X_hist', fill: '#ede9fe' },
      ];
      chips.forEach((chip) => drawPill(ctx, chip.rect, chip.label, chip.fill, '#334155', scale, 14));
      strokePolyline(ctx, [rectTop(chips[0].rect), rectLeft(boxes.inputEmbedding, 0.22)], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
      strokePolyline(ctx, [rectTop(chips[1].rect), rectLeft(boxes.inputEmbedding, 0.72)], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
      strokePolyline(ctx, [rectTop(chips[2].rect), rectLeft(boxes.outputEmbedding, 0.3)], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
      strokePolyline(ctx, [rectTop(chips[3].rect), rectLeft(boxes.outputEmbedding, 0.78)], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
      drawInfoCard(ctx, createRect(layout.frameEncoder.x - 18 * scale, layout.frameEncoder.y - 108 * scale, 262 * scale, 88 * scale), '原始输入拆分', [
        'RP / Trip -> X_rp [B,d_rp]',
        '环境统计 -> X_env [B,d_env]',
        '候选与上下文 -> X_cand, X_hist',
      ], scale, '#38bdf8');
    } else if (demoFlowState.step === 'concat') {
      const strip = createRect(layout.frameEncoder.x + 26 * scale, layout.frameEncoder.y - 96 * scale, 430 * scale, 58 * scale);
      drawRoundedRect(ctx, strip, 16 * scale, 'rgba(15,23,42,0.06)', '#1f2937', 2.2 * scale);
      const tokenLabels = ['E_rp', 'E_env', 'E_hist,1', 'E_hist,2', 'E_cand,1'];
      tokenLabels.forEach((label, idx) => {
        const token = createRect(strip.x + 16 * scale + idx * 80 * scale, strip.y + 12 * scale, 70 * scale, 34 * scale);
        drawPill(ctx, token, label, '#ffffff', '#5b8ef7', scale, 13);
      });
      drawInfoCard(ctx, createRect(strip.x + 460 * scale, strip.y - 2 * scale, 182 * scale, 60 * scale), 'Concat', [
        'X_enc [B,L_enc,d_model]',
      ], scale, '#0f172a');
      strokePolyline(ctx, [rectTop(strip), rectBottom(boxes.encoder)], { color: DIAGRAM_LINE_COLOR, width: 2.4 * scale, arrow: true });
    } else if (demoFlowState.step === 'headDetail') {
      const headRect = boxes[demoFlowState.headFocus] || boxes.maskHead;
      const panel = createRect(headRect.x - 180 * scale, headRect.y + 88 * scale, 170 * scale, 138 * scale);
      drawNeuralPanel(ctx, panel, scale, `${demoFlowState.headFocus} 前向传播`, '#2563eb');
      strokePolyline(ctx, [rectLeft(headRect, 0.8), rectRight(panel, 0.5)], { color: DIAGRAM_LINE_COLOR, width: 2.5 * scale, arrow: true });
    }
  }

  if (demoFlowState.mode === 'attention') {
    const matrix = createRect(layout.frameEncoder.x - 64 * scale, layout.frameEncoder.y - 150 * scale, 286 * scale, 122 * scale);
    drawInfoCard(ctx, matrix, `多份数据并行输入 B=${demoFlowState.sampleCount}`, [
      '样本1: [RP][env][ctx][cand]',
      '样本2: [RP][env][ctx][cand]',
      '样本3: [RP][env][ctx][cand]',
    ], scale, '#0ea5e9');
    const q = createRect(layout.frameEncoder.x + 250 * scale, layout.frameEncoder.y - 152 * scale, 98 * scale, 54 * scale);
    const k = createRect(layout.frameEncoder.x + 360 * scale, layout.frameEncoder.y - 152 * scale, 98 * scale, 54 * scale);
    const v = createRect(layout.frameEncoder.x + 470 * scale, layout.frameEncoder.y - 152 * scale, 98 * scale, 54 * scale);
    drawPill(ctx, q, 'Q', '#dbeafe', '#2563eb', scale, 18);
    drawPill(ctx, k, 'K', '#dcfce7', '#16a34a', scale, 18);
    drawPill(ctx, v, 'V', '#fef3c7', '#ca8a04', scale, 18);
    drawInfoCard(ctx, createRect(layout.frameDecoder.x + 140 * scale, layout.frameEncoder.y - 150 * scale, 206 * scale, 120 * scale), 'Attention 计算', [
      'S = QK^T / sqrt(d_h)',
      'A = softmax(S)',
      'O = A V',
      'concat(O) -> H',
    ], scale, '#8b5cf6');
  }

  if (demoFlowState.mode === 'training') {
    const losses = [
      ['L_count', '#ef4444'],
      ['L_slot', '#fbbf24'],
      ['L_mask', '#f59e0b'],
      ['L_value', '#10b981'],
      ['L_score', '#8b5cf6'],
    ];
    losses.forEach(([label, accent], idx) => {
      const row = idx < 3 ? 0 : 1;
      const col = idx < 3 ? idx : idx - 3;
      const rect = createRect(layout.frameHeads.x + 4 * scale + col * 98 * scale, layout.frameHeads.y + 148 * scale + row * 70 * scale, 92 * scale, 54 * scale);
      drawPill(ctx, rect, label, '#ffffff', accent, scale, 15);
    });
    drawInfoCard(ctx, createRect(layout.frameHeads.x + 10 * scale, layout.frameHeads.y + 300 * scale, 290 * scale, 84 * scale), '总损失', [
      'L_total = λcLc + λsLs + λmLm + λvLv + λqLq',
      '约束惩罚 / reward 也在此汇总',
    ], scale, '#0f172a');
    strokePolyline(ctx, [rectBottom(boxes.maskHead), rectTop(boxes.decoderAddNormTop)], {
      color: 'rgba(239,68,68,0.8)',
      width: 2.4 * scale,
      dash: [8 * scale, 6 * scale],
      arrow: true,
    });
  }

  if (demoFlowState.mode === 'dispatch') {
    const decision = createRect(boxes.countHead.x + 132 * scale, boxes.countHead.y + 2 * scale, 150 * scale, 60 * scale);
    const chooseEight = demoFlowState.countDecision === 'eight';
    drawInfoCard(ctx, decision, '题数判定', [
      chooseEight ? 'Count Head -> 8 题' : 'Count Head -> 12 题',
      chooseEight ? 'slot_select_head 选中 8 个题位' : 'slot_select_head 选中 12 个题位',
    ], scale, chooseEight ? '#ef4444' : '#16a34a');
    strokePolyline(ctx, [rectRight(boxes.countHead, 0.5), rectLeft(decision, 0.5)], { color: chooseEight ? '#ef4444' : '#16a34a', width: 2.4 * scale, arrow: true });
    strokePolyline(ctx, [rectBottom(boxes.outputBlock), rectTop(layout.outputSourceA)], { color: DIAGRAM_LINE_COLOR, width: 2.2 * scale, arrow: true });
    strokePolyline(ctx, [rectLeft(layout.outputSourceA, 0.8), { x: layout.inputSource.x + layout.inputSource.width / 2, y: layout.inputSource.y + 18 * scale }], {
      color: '#64748b',
      width: 2.1 * scale,
      dash: [7 * scale, 6 * scale],
      arrow: true,
    });
    drawInfoCard(ctx, createRect(layout.inputSource.x + 4 * scale, layout.inputSource.y - 154 * scale, 184 * scale, 56 * scale), '下一位 respondent', [
      '重新填写 RP / Trip',
    ], scale, '#334155');
  }

  if (progress < 1) {
    ctx.save();
    ctx.globalAlpha = 0.22 * (1 - progress);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, layout.width || 0, layout.height || 0);
    ctx.restore();
  }
}

function createTextSprite(text, {
  width = 520,
  height = 150,
  fontSize = 40,
  scale = 0.45,
  fill = '#e5eefc',
  background = 'rgba(8,14,24,0.65)',
  border = 'rgba(97,218,251,0.45)',
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  material.depthTest = false;
  material.depthWrite = false;
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width * scale, height * scale, 1);

  function draw(nextText, overrides = {}) {
    const resolvedFontSize = overrides.fontSize ?? fontSize;
    const resolvedFill = overrides.fill ?? fill;
    const resolvedBackground = overrides.background ?? background;
    const resolvedBorder = overrides.border ?? border;
    const lines = String(nextText ?? '').split('\n');
    const lineHeight = resolvedFontSize * 1.28;
    const totalHeight = lineHeight * lines.length;
    const startY = (height - totalHeight) / 2 + resolvedFontSize * 0.7;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = resolvedBackground;
    ctx.strokeStyle = resolvedBorder;
    ctx.lineWidth = 4;
    const radius = 24;
    ctx.beginPath();
    ctx.moveTo(radius, 0);
    ctx.lineTo(width - radius, 0);
    ctx.quadraticCurveTo(width, 0, width, radius);
    ctx.lineTo(width, height - radius);
    ctx.quadraticCurveTo(width, height, width - radius, height);
    ctx.lineTo(radius, height);
    ctx.quadraticCurveTo(0, height, 0, height - radius);
    ctx.lineTo(0, radius);
    ctx.quadraticCurveTo(0, 0, radius, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = resolvedFill;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `700 ${resolvedFontSize}px PingFang SC, Arial, sans-serif`;
    lines.forEach((line, index) => {
      ctx.fillText(line, width / 2, startY + index * lineHeight);
    });
    texture.needsUpdate = true;
  }

  draw(text);
  sprite.userData.updateText = draw;
  return sprite;
}

function createLine(points, color, opacity = 0.28) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  return new THREE.Line(geometry, material);
}

function addEmbeddingVisual(group, width, height, color) {
  const sub = new THREE.Group();
  const cols = 5;
  const rows = 4;
  for (let x = 0; x < cols; x += 1) {
    for (let y = 0; y < rows; y += 1) {
      const box = new THREE.Mesh(
        new THREE.BoxGeometry(10, 10 + ((x + y) % 3) * 6, 8),
        new THREE.MeshPhongMaterial({ color: lerpColorHex(color, '#ffffff', 0.22), transparent: true, opacity: 0.82 }),
      );
      box.position.set(
        -width * 0.22 + x * 20,
        -height * 0.18 + y * 18,
        4,
      );
      sub.add(box);
    }
  }
  group.add(sub);
  return sub;
}

function addWaveVisual(group, width, height, color) {
  const points = [];
  for (let i = 0; i <= 40; i += 1) {
    const t = i / 40;
    points.push(new THREE.Vector3(
      -width * 0.3 + t * width * 0.6,
      Math.sin(t * Math.PI * 4) * height * 0.16,
      4,
    ));
  }
  const line = createLine(points, color, 0.86);
  group.add(line);
  return line;
}

function addNetworkVisual(group, width, height, color, layerConfig = [4, 6, 4]) {
  const nodeMaterial = new THREE.MeshPhongMaterial({ color: lerpColorHex(color, '#ffffff', 0.35), emissive: 0x0, transparent: true, opacity: 0.94 });
  const lineMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.46 });
  const nodeGeo = new THREE.SphereGeometry(5.5, 18, 18);
  const layers = [];

  layerConfig.forEach((count, idx) => {
    const x = -width * 0.26 + idx * (width * 0.26);
    const nodes = [];
    for (let i = 0; i < count; i += 1) {
      const y = (i - (count - 1) / 2) * 18;
      const node = new THREE.Mesh(nodeGeo, nodeMaterial.clone());
      node.position.set(x, y, 8);
      group.add(node);
      nodes.push(node);
    }
    layers.push(nodes);
  });

  for (let i = 0; i < layers.length - 1; i += 1) {
    layers[i].forEach((a) => {
      layers[i + 1].forEach((b) => {
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([a.position.clone(), b.position.clone()]),
          lineMaterial.clone(),
        );
        group.add(line);
      });
    });
  }

  for (let i = 0; i < 3; i += 1) {
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(width * 0.78, height * 0.72, 6),
      new THREE.MeshPhongMaterial({ color: lerpColorHex(color, '#ffffff', 0.14), transparent: true, opacity: 0.12 }),
    );
    plate.position.set(0, 0, -16 - i * 10);
    group.add(plate);
  }
}

function addAttentionVisual(group, width, height, color) {
  const colors = ['#61dafb', '#7ee787', '#fbbf24', '#fb7185'];
  for (let i = 0; i < 4; i += 1) {
    const head = new THREE.Mesh(
      new THREE.BoxGeometry(width * 0.16, height * 0.34, 10),
      new THREE.MeshPhongMaterial({ color: colors[i], transparent: true, opacity: 0.64 }),
    );
    head.position.set(-width * 0.24 + i * width * 0.16, 0, 8);
    group.add(head);
  }
  const slash = createLine([
    new THREE.Vector3(-width * 0.32, height * 0.18, 18),
    new THREE.Vector3(width * 0.32, -height * 0.18, 18),
  ], color, 0.8);
  group.add(slash);
}

function addMultiHeadAttentionVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const tint = visualSpec.tint || spec.haloColor || '#fbbf24';
  const fixedZ = 30;
  const frameWidth = spec.size.x * 0.9;
  const frameHeight = spec.size.y * 0.92;
  const frameTopY = frameHeight / 2;
  const frameBottomY = -frameHeight / 2;
  const activeObjects = [];
  const concatBatchShift = { x: 0, y: -7.2, z: 0 };
  const attentionBatchShift = { x: 0, y: -3.4, z: 0 };
  const batchRemapObjects = [];
  const headSplitObjects = [];

  const cubeMaterial = (rawColor, opacity = 0.68) => new THREE.MeshPhongMaterial({
    color: rawColor,
    transparent: true,
    opacity,
    emissive: new THREE.Color(rawColor),
    emissiveIntensity: 0.08,
    shininess: 68,
  });

  const addFormulaAt = (formulaText, x, y, {
    width = 500,
    height = 48,
    fontSize = 12,
    fill = '#e2e8f0',
    scale = 0.052,
    align = 'center',
    z = fixedZ + 6,
  } = {}) => {
    const sprite = createFormulaSprite(formulaText, {
      width,
      height,
      fontSize,
      scale,
      align,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, z);
    interior.add(sprite);
    return sprite;
  };

  const addPlainLabel = (text, x, y, z = fixedZ + 7, {
    width = 150,
    height = 34,
    fontSize = 14,
    fill = '#e2e8f0',
    scale = 0.064,
  } = {}) => {
    const label = createOverlaySprite(text, {
      width,
      height,
      fontSize,
      scale,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    label.position.set(x, y, z);
    interior.add(label);
    return label;
  };

  const addPlaneMatrix = ({ label, x, y, rows = 2, cols = 3, color = '#93c5fd', cell = 5.8, opacity = 0.68 }) => {
    const local = [];
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const cube = new THREE.Mesh(
          new THREE.BoxGeometry(cell, cell, cell),
          cubeMaterial(color, opacity),
        );
        cube.position.set(x + (c - (cols - 1) / 2) * cell * 1.28, y - r * cell * 1.28, fixedZ);
        interior.add(cube);
        local.push(cube);
        activeObjects.push(cube);
      }
    }
    addPlainLabel(label, x, y + cell * 1.55, fixedZ + 6, { width: 120, fontSize: 13, scale: 0.06 });
    return { cubes: local, center: new THREE.Vector3(x, y - (rows - 1) * cell * 0.64, fixedZ) };
  };

  const addTensorMatrix = ({
    label,
    x,
    y,
    z = fixedZ + 1,
    lCols = 4,
    dRows = 3,
    bLayers = 3,
    color = '#93c5fd',
    segments = null,
    segmentAxis = 'l',
    cell = 4.4,
    opacity = 0.5,
    batchOpacityStep = 0.16,
    batchShiftX = 0,
    batchShiftY = -5.0,
    batchShiftZ = -2.1,
    xSpacingFactor = 1.18,
    zSpacingFactor = 1.18,
    lGroupSize = 0,
    lGroupGap = 0,
    nodeShape = 'box',
    labelOptions = {},
    tensorInfo = {},
  }) => {
    const local = [];
    const cells = [];
    const weightSources = [];
    const frontLayerPoints = [];
    const frontLayerColoredPoints = [];
    const normalizedSegments = Array.isArray(segments) && segments.length ? segments : null;
    const colorForAxisIndex = (axisIndex) => {
      if (!normalizedSegments) return color;
      let cursor = 0;
      for (const segment of normalizedSegments) {
        const count = Math.max(1, Number(segment.count || 1));
        if (axisIndex >= cursor && axisIndex < cursor + count) {
          return segment.color || color;
        }
        cursor += count;
      }
      return normalizedSegments[normalizedSegments.length - 1].color || color;
    };
    const groupCount = lGroupSize > 0 ? Math.ceil(lCols / lGroupSize) : 1;
    const lOffset = (lIndex) => {
      if (!(lGroupSize > 0) || !lGroupGap) return 0;
      const groupIndex = Math.floor(lIndex / lGroupSize);
      return (groupIndex - (groupCount - 1) / 2) * lGroupGap;
    };
    const xAtL = (lIndex) => (
      x + (lIndex - (lCols - 1) / 2) * cell * xSpacingFactor + lOffset(lIndex)
    );

    for (let b = 0; b < bLayers; b += 1) {
      for (let l = 0; l < lCols; l += 1) {
        for (let d = 0; d < dRows; d += 1) {
          const cubeColor = colorForAxisIndex(segmentAxis === 'd' ? d : l);
          const geometry = nodeShape === 'sphere'
            ? new THREE.SphereGeometry(cell * 0.54, 16, 12)
            : new THREE.BoxGeometry(cell, cell, cell);
          const cube = new THREE.Mesh(
            geometry,
            cubeMaterial(cubeColor, Math.max(0.08, opacity - b * batchOpacityStep)),
          );
          const basePosition = new THREE.Vector3(
            xAtL(l),
            y,
            z - d * cell * zSpacingFactor,
          );
          cube.position.copy(basePosition).add(new THREE.Vector3(
            b * batchShiftX,
            b * batchShiftY,
            b * batchShiftZ,
          ));
          interior.add(cube);
          local.push(cube);
          const isLastTensorCell = b === bLayers - 1 && l === lCols - 1 && d === dRows - 1;
          cube.userData.attentionTensorNode = {
            moduleId: tensorInfo.moduleId || moduleEntry.id || 'encoder',
            tensorKey: tensorInfo.key || label,
            tensorLabel: tensorInfo.label || label,
            tensorShapeFormula: tensorInfo.shapeFormula || `${label} \\in \\mathbb{R}^{B\\times L\\times d}`,
            tensorStage: tensorInfo.stage || 'Multi-Head Attention Tensor',
            bIndex: b + 1,
            lIndex: l + 1,
            dIndex: d + 1,
            bLayers,
            lCols,
            dRows,
            bSymbol: tensorInfo.bSymbol || 'B',
            lSymbol: tensorInfo.lSymbol || 'L',
            dSymbol: tensorInfo.dSymbol || 'd',
            isLastTensorCell,
            color: cubeColor,
          };
          attentionTensorTargets.push(cube);
          cells.push({
            mesh: cube,
            b,
            l,
            d,
            color: cubeColor,
            basePosition,
            position: cube.position.clone(),
          });
            if (b === 0) {
              const frontPoint = cube.position.clone();
              activeObjects.push(cube);
              frontLayerPoints.push(frontPoint);
              frontLayerColoredPoints.push({
                point: frontPoint,
                color: cubeColor,
                tokenIndex: l,
                featureIndex: d,
              });
            }
          weightSources.push({
            point: cube.position.clone().add(new THREE.Vector3(0, cell * 0.6, 0)),
            color: cubeColor,
          });
        }
      }
    }
    if (normalizedSegments) {
      let cursor = 0;
      normalizedSegments.forEach((segment) => {
        const count = Math.max(1, Number(segment.count || 1));
        const centerCol = cursor + (count - 1) / 2;
        const segmentX = segmentAxis === 'd'
          ? x
          : xAtL(centerCol);
        const segmentZ = segmentAxis === 'd'
          ? z - centerCol * cell * zSpacingFactor
          : z + 2;
        addPlainLabel(segment.label || segment.key || '', segmentX, y - bLayers * Math.abs(batchShiftY) - cell * 1.9, segmentZ, {
          width: 112,
          height: 30,
          fontSize: 11,
          fill: segment.color || '#e2e8f0',
          scale: 0.046,
        });
        cursor += count;
      });
    }
    addPlainLabel(label, x, y + cell * 1.95, z + 2, {
      width: 148,
      fontSize: 13,
      scale: 0.058,
      ...labelOptions,
    });
    return {
      cubes: local,
      cells,
      weightSources,
      frontLayerPoints,
      frontLayerColoredPoints,
      batchShift: new THREE.Vector3(batchShiftX, batchShiftY, batchShiftZ),
      center: new THREE.Vector3(x, y - (bLayers - 1) * Math.abs(batchShiftY) * 0.28, z - (dRows - 1) * cell * zSpacingFactor * 0.58),
      top: new THREE.Vector3(x, y + cell * 1.0, z),
      bottom: new THREE.Vector3(x, y - (bLayers - 1) * Math.abs(batchShiftY) - cell, z - (dRows - 1) * cell * zSpacingFactor * 0.58),
    };
  };

  const addBatchAxisRemapAnimation = (tensor, {
    label = 'concat B copies -> attention B stack',
    color = '#fef3c7',
  } = {}) => {
    if (!tensor?.cells?.length) return;
    const concatEntry = moduleMap.get('concatPanel');
    const sourceCells = concatEntry?.concatRemapCells || [];
    const sourceByIndex = new Map(sourceCells.map((cellInfo) => [
      `${cellInfo.b}|${cellInfo.l}|${cellInfo.d}`,
      cellInfo,
    ]));
    const maxL = Math.max(...tensor.cells.map((cellInfo) => cellInfo.l));
    const maxD = Math.max(...tensor.cells.map((cellInfo) => cellInfo.d));
    const maxB = Math.max(...tensor.cells.map((cellInfo) => cellInfo.b));
    const moverSize = 2.25;
    const layerGap = maxB > 0 ? 0.31 : 0;

    tensor.cells.forEach((cellInfo) => {
      // b=0 来自 concat 正面矩阵；b=1/2 来自 concat 后侧 -x/-z 的 ghost copy。
      const sourceCell = sourceByIndex.get(`${cellInfo.b}|${cellInfo.l}|${cellInfo.d}`);
      const start = sourceCell?.mesh && concatEntry?.group
        ? group.worldToLocal(concatEntry.group.localToWorld(sourceCell.mesh.position.clone()))
        : cellInfo.basePosition.clone().add(new THREE.Vector3(
          concatBatchShift.x * cellInfo.b,
          concatBatchShift.y * cellInfo.b,
          concatBatchShift.z * cellInfo.b,
        ));
      const end = cellInfo.position.clone();
      const mover = new THREE.Mesh(
        new THREE.BoxGeometry(moverSize, moverSize, moverSize),
        cubeMaterial(cellInfo.color || color, cellInfo.b === 0 ? 0.34 : 0.48),
      );
      mover.position.copy(start);
      mover.renderOrder = 70;
      interior.add(mover);

      const isTrailCell = cellInfo.b > 0
        && (cellInfo.l === 0 || cellInfo.l === maxL)
        && (cellInfo.d === 0 || cellInfo.d === maxD);
      if (isTrailCell) {
        interior.add(createLine([start, end], cellInfo.color || color, 0.12));
      }

      batchRemapObjects.push({
        mesh: mover,
        start,
        end,
        baseOpacity: cellInfo.b === 0 ? 0.34 : 0.48,
        layer: cellInfo.b,
        phase: cellInfo.b * layerGap,
      });
    });

    addPlainLabel(label, -frameWidth * 0.32, frameBottomY + 36, fixedZ + 8, {
      width: 290,
      height: 30,
      fontSize: 11,
      fill: color,
      scale: 0.044,
    });
  };

    const addDenseConnections = (fromPoints, toPoints, color, opacity = 0.055) => {
      const positions = new Float32Array(fromPoints.length * toPoints.length * 2 * 3);
      let cursor = 0;
      fromPoints.forEach((from) => {
        toPoints.forEach((to) => {
          positions[cursor++] = from.x;
          positions[cursor++] = from.y;
          positions[cursor++] = from.z;
          positions[cursor++] = to.x;
          positions[cursor++] = to.y;
          positions[cursor++] = to.z;
        });
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      });
      const segments = new THREE.LineSegments(geometry, material);
      interior.add(segments);
      return segments;
    };

    const addSourceColoredDenseConnections = (fromEntries, toPoints, opacity = 0.055) => {
      const grouped = new Map();
      fromEntries.forEach((entry) => {
        const color = entry.color || '#c4b5fd';
        if (!grouped.has(color)) grouped.set(color, []);
        grouped.get(color).push(entry.point || entry);
      });
    grouped.forEach((points, color) => {
      addDenseConnections(points, toPoints, color, opacity);
    });
  };

  const addSourceColoredCellConnections = (fromCells, toCells, opacity = 0.025) => {
    const grouped = new Map();
    fromCells.forEach((entry) => {
      const color = entry.color || '#c4b5fd';
      if (!grouped.has(color)) grouped.set(color, []);
      grouped.get(color).push(entry.position || entry.point);
    });
    const toPoints = toCells.map((entry) => entry.position || entry.point).filter(Boolean);
    grouped.forEach((points, color) => {
      addDenseConnections(points.filter(Boolean), toPoints, color, opacity);
    });
  };

    const addProjectionNet = ({
    label,
    x,
    y,
    color = '#93c5fd',
    inputCount = 5,
    outputCount = 5,
    inputDim = 'n_{in}=d_{model}',
    outputDim = 'n_{out}=h d_h',
  }) => {
    const net = new THREE.Group();
    const nodeMaterial = cubeMaterial(color, 0.86);
    const edgeMaterial = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.34,
    });
    const inputNodes = [];
    const outputNodes = [];
    const nodeGap = 5.6;
    const inputStartX = -(inputCount - 1) * nodeGap / 2;
    const outputStartX = -(outputCount - 1) * nodeGap / 2;
    const inputY = y - 7.6;
    const outputY = y + 7.6;

    for (let idx = 0; idx < inputCount; idx += 1) {
      const node = new THREE.Mesh(new THREE.SphereGeometry(2.45, 14, 10), nodeMaterial);
      node.position.set(x + inputStartX + idx * nodeGap, inputY, fixedZ + 1);
      net.add(node);
      inputNodes.push(node);
      activeObjects.push(node);
    }
    for (let idx = 0; idx < outputCount; idx += 1) {
      const node = new THREE.Mesh(new THREE.SphereGeometry(2.15, 14, 10), nodeMaterial);
      node.position.set(x + outputStartX + idx * nodeGap, outputY, fixedZ + 1);
      net.add(node);
      outputNodes.push(node);
      activeObjects.push(node);
    }
    inputNodes.forEach((from) => {
      outputNodes.forEach((to) => {
        const edge = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([from.position.clone(), to.position.clone()]),
          edgeMaterial,
        );
        net.add(edge);
      });
    });

    const matrixPlate = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(inputCount, outputCount) * nodeGap + 5, 3.8, 1.4),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: 0.18,
        emissive: new THREE.Color(color),
        emissiveIntensity: 0.06,
        shininess: 40,
      }),
    );
    matrixPlate.position.set(x, y, fixedZ);
    net.add(matrixPlate);

    const tag = createOverlaySprite(`${label} linear`, {
      width: 146,
      height: 32,
      fontSize: 12,
      scale: 0.055,
      fill: '#f8fafc',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    tag.position.set(x, y + 14, fixedZ + 5);
    net.add(tag);

    const dimTag = createFormulaSprite(`${label}\\in\\mathbb{R}^{n_{in}\\times n_{out}}`, {
      width: 260,
      height: 34,
      fontSize: 11,
      scale: 0.042,
      align: 'center',
      fill: '#fde68a',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    dimTag.position.set(x, y - 16, fixedZ + 5);
    net.add(dimTag);

    addPlainLabel(inputDim, x - Math.max(inputCount, outputCount) * nodeGap * 0.55 - 6, inputY, fixedZ + 5, {
      width: 150,
      height: 28,
      fontSize: 10,
      fill: '#bfdbfe',
      scale: 0.042,
    });
    addPlainLabel(outputDim, x + Math.max(inputCount, outputCount) * nodeGap * 0.55 + 6, outputY, fixedZ + 5, {
      width: 150,
      height: 28,
      fontSize: 10,
      fill: '#bfdbfe',
      scale: 0.042,
    });

    interior.add(net);
    return {
      center: new THREE.Vector3(x, y, fixedZ + 1),
      bottom: new THREE.Vector3(x, inputY, fixedZ + 1),
      top: new THREE.Vector3(x, outputY, fixedZ + 1),
      inputAnchors: inputNodes.map((node) => node.position.clone()),
      outputAnchors: outputNodes.map((node) => node.position.clone()),
    };
  };

  const addOperationBox = ({ label, x, y, color = '#fde68a', width = 52, height = 16, opacity = 0.52 }) => {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, 7),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity,
        emissive: new THREE.Color(color),
        emissiveIntensity: 0.08,
        shininess: 60,
      }),
    );
    box.position.set(x, y, fixedZ);
    interior.add(box);
    activeObjects.push(box);
    addPlainLabel(label, x, y, fixedZ + 6, { width: 170, fontSize: 13, scale: 0.06, fill: '#111827' });
    return { mesh: box, center: box.position.clone(), bottom: new THREE.Vector3(x, y - height * 0.55, fixedZ), top: new THREE.Vector3(x, y + height * 0.55, fixedZ) };
  };

  const makeHeadFrame = ({ dx = 0, dz = 0, opacity = 0.18, label = 'Head 2', active = false } = {}) => {
    const x0 = -frameWidth * 0.48 + dx;
    const x1 = frameWidth * 0.48 + dx;
    const y0 = frameTopY - 50;
    const y1 = frameTopY - 2;
    const z = fixedZ + dz;
    const points = [
      new THREE.Vector3(x0, y1, z),
      new THREE.Vector3(x1, y1, z),
      new THREE.Vector3(x1, y0, z),
      new THREE.Vector3(x0, y0, z),
      new THREE.Vector3(x0, y1, z),
    ];
    interior.add(createLine(points, active ? '#fef3c7' : tint, opacity));
    addPlainLabel(label, x0 + 22, y1 + 5, z + 5, {
      width: 130,
      fontSize: active ? 14 : 12,
      scale: 0.055,
      fill: active ? '#fef3c7' : '#cbd5e1',
    });
  };

  const addWireBox = ({
    label,
    x,
    y,
    z,
    width,
    height,
    depth,
    color = '#fef3c7',
    opacity = 0.5,
    showCenterLabel = true,
  }) => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, depth),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.035,
        depthWrite: false,
      }),
    );
    mesh.position.set(x, y, z);
    interior.add(mesh);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      }),
    );
    edges.position.copy(mesh.position);
    interior.add(edges);
    const labelSprite = showCenterLabel ? addPlainLabel(label, x, y + height * 0.62, z + depth * 0.5 + 2, {
      width: 130,
      height: 28,
      fontSize: 11,
      fill: color,
      scale: 0.048,
    }) : null;
    return {
      mesh,
      edges,
      labelSprite,
      center: mesh.position.clone(),
      top: new THREE.Vector3(x, y + height * 0.5, z),
      bottom: new THREE.Vector3(x, y - height * 0.5, z),
    };
  };

  const getTightCellBounds = (cells, {
    marginX = 2.4,
    marginY = 2.4,
    marginZ = 2.6,
  } = {}) => {
    const xs = cells.map((entry) => entry.position.x);
    const ys = cells.map((entry) => entry.position.y);
    const zs = cells.map((entry) => entry.position.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    return {
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      centerZ: (minZ + maxZ) / 2,
      width: (maxX - minX) + marginX,
      height: (maxY - minY) + marginY,
      depth: (maxZ - minZ) + marginZ,
    };
  };

  const addHeadSplitVisual = ({
    x,
    y,
    z,
    width,
    height,
    depth,
    heads = 3,
    totalL = 6,
    prefix = 'Head',
    bLayers = 3,
    dRows = 12,
    baseColor = '#93c5fd',
    panelTitle = 'Head',
    inputFormula = 'X \\in \\mathbb{R}^{B\\times L\\times d}',
    outputFormula = 'X_i \\in \\mathbb{R}^{B\\times (L/h)\\times d}',
  }) => {
    const boxes = [];
    const perHead = Math.ceil(totalL / heads);
    const startGapX = width * 0.28;
    const targetGapX = width * 0.42;
    const boxWidth = Math.max(20, width * 0.34);
    const boxDepth = depth * 0.62;
    for (let head = 0; head < heads; head += 1) {
      const lStart = head * perHead + 1;
      const lEnd = Math.min(totalL, (head + 1) * perHead);
      const headColor = baseColor;
      const startX = x + (head - (heads - 1) / 2) * startGapX;
      const targetX = x + (head - (heads - 1) / 2) * targetGapX;
      const box = addWireBox({
        label: `${prefix}${head + 1}:L${lStart}-${lEnd}`,
        x: startX,
        y,
        z,
        width: boxWidth,
        height,
        depth: boxDepth,
        color: headColor,
        opacity: Math.max(0.18, 0.74 - head * 0.20),
        showCenterLabel: false,
      });
      const interiorGroup = new THREE.Group();
      interiorGroup.position.set(startX, y, z);
      interior.add(interiorGroup);
      const headL = Math.max(1, lEnd - lStart + 1);
      const innerUsableWidth = boxWidth * 0.76;
      const innerUsableHeight = height * 0.52;
      const innerUsableDepth = boxDepth * 0.68;
      const innerStepX = headL > 1 ? innerUsableWidth / (headL - 1) : 0;
      const innerStepY = bLayers > 1 ? innerUsableHeight / (bLayers - 1) : 0;
      const innerStepZ = dRows > 1 ? innerUsableDepth / (dRows - 1) : 0;
      const innerRadius = Math.min(
        0.88,
        Math.max(0.42, innerStepX * 0.28),
        Math.max(0.42, innerStepY * 0.24),
        Math.max(0.42, innerStepZ * 0.26),
      );
      const innerCellTargets = [];
      const targetByKey = new Map();
      for (let b = 0; b < bLayers; b += 1) {
        for (let lLocal = 0; lLocal < headL; lLocal += 1) {
          for (let d = 0; d < dRows; d += 1) {
            const relX = (lLocal - (headL - 1) / 2) * innerStepX;
            const relY = ((bLayers - 1) / 2 - b) * innerStepY - 0.4;
            const relZ = -d * innerStepZ;
            const innerCell = new THREE.Mesh(
              new THREE.SphereGeometry(innerRadius, 12, 10),
              cubeMaterial(headColor, 0.22),
            );
            innerCell.position.set(relX, relY, relZ);
            interiorGroup.add(innerCell);
            const targetInfo = {
              mesh: innerCell,
              b,
              lLocal,
              d,
              relX,
              relY,
              relZ,
              target: new THREE.Vector3(targetX + relX, y + relY, z + relZ),
            };
            innerCellTargets.push(targetInfo);
            targetByKey.set(`${b}|${lLocal}|${d}`, targetInfo);
          }
        }
      }
      box.mesh.material.opacity = Math.max(0.01, 0.05 - head * 0.012);
      box.edges.material.opacity = Math.max(0.18, 0.74 - head * 0.20);
      box.mesh.position.x = startX;
      box.edges.position.x = startX;
      headSplitObjects.push({
        mesh: box.mesh,
        edges: box.edges,
        labelSprite: box.labelSprite,
        interiorGroup,
        innerCellTargets,
        headIndex: head,
        startX,
        targetX,
        fixedZ: z,
        baseOpacity: box.edges.material.opacity,
        phase: head * 0.18,
      });
      boxes.push({
        ...box,
        headIndex: head,
        lStart,
        lEnd,
        headColor,
        interiorGroup,
        headL,
        bLayers,
        dRows,
        innerCellTargets,
        targetByKey,
        startX,
        targetX,
      });
    }
    return {
      boxes,
      center: new THREE.Vector3(x, y, z),
      top: new THREE.Vector3(x, y + height * 0.5, z),
    };
  };

  const addInlineHeadSplitFrames = ({
    tensorLabel = 'Q',
    sourceTensor,
    color = '#93c5fd',
    totalL = 12,
    heads = 3,
  }) => {
    const items = [];
    const perHeadL = Math.ceil(totalL / heads);
    const totalFormula = `${tensorLabel}\\in\\mathbb{R}^{B\\times L\\times d}`;
    const outputFormula = `${tensorLabel}_i\\in\\mathbb{R}^{B\\times (L/h)\\times d}`;

    for (let head = 0; head < heads; head += 1) {
      const lStart = head * perHeadL;
      const lEnd = Math.min(totalL - 1, lStart + perHeadL - 1);
      const sourceSlice = (sourceTensor?.cells || []).filter((entry) => entry.l >= lStart && entry.l <= lEnd);
      const bounds = getTightCellBounds(sourceSlice, {
        marginX: 2.2,
        marginY: 2.1,
        marginZ: 2.3,
      });
      const box = addWireBox({
        label: '',
        x: bounds.centerX,
        y: bounds.centerY,
        z: bounds.centerZ,
        width: bounds.width,
        height: bounds.height,
        depth: bounds.depth,
        color,
        opacity: Math.max(0.18, 0.74 - head * 0.16),
        showCenterLabel: false,
      });
      const headLabel = createOverlaySprite(`head-${head + 1}`, {
        width: 86,
        height: 26,
        fontSize: 12,
        scale: 0.046,
        fill: color,
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      headLabel.position.set(bounds.centerX - bounds.width * 0.26, bounds.centerY + bounds.height * 0.38, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(headLabel);

      const inputLabel = createFormulaSprite(totalFormula, {
        width: 220,
        height: 28,
        fontSize: 10,
        scale: 0.036,
        align: 'left',
        fill: '#dbeafe',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      inputLabel.position.set(bounds.centerX - bounds.width * 0.12, bounds.centerY - bounds.height * 0.40, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(inputLabel);

      const outputLabel = createFormulaSprite(outputFormula, {
        width: 220,
        height: 28,
        fontSize: 10,
        scale: 0.036,
        align: 'right',
        fill: '#fef3c7',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      outputLabel.position.set(bounds.centerX + bounds.width * 0.12, bounds.centerY + bounds.height * 0.38, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(outputLabel);

      items.push({
        box,
        headLabel,
        inputLabel,
        outputLabel,
        headIndex: head,
        lStart,
        lEnd,
        cells: sourceSlice,
        centerX: bounds.centerX,
        top: new THREE.Vector3(bounds.centerX, bounds.centerY + bounds.height * 0.5, bounds.centerZ),
        bottom: new THREE.Vector3(bounds.centerX, bounds.centerY - bounds.height * 0.5, bounds.centerZ),
      });
    }
    return { items };
  };

  const addHeadTensorRow = ({
    prefix,
    centers,
    y,
    z,
    lCols,
    dRows,
    bLayers = 3,
    color = '#93c5fd',
    shapeFormula = 'T \\in \\mathbb{R}^{B\\times L\\times d}',
    stage = 'Head Tensor',
    totalFormula = null,
    outputFormula = null,
    dSymbol = 'd',
    labelFill = '#f8fafc',
  }) => {
    const items = [];
    const cell = 2.45;
    const xSpacingFactor = 1.28;
    const zSpacingFactor = 1.24;
    const batchShiftY = attentionBatchShift.y * 0.86;
    centers.forEach((centerX, head) => {
      const tensor = addTensorMatrix({
        label: `${prefix}${head + 1}`,
        x: centerX,
        y,
        z,
        lCols,
        dRows,
        bLayers,
        color,
        cell,
        opacity: 0.34,
        batchShiftX: 0,
        batchShiftY,
        batchShiftZ: 0,
        xSpacingFactor,
        zSpacingFactor,
        nodeShape: 'sphere',
        tensorInfo: {
          key: `${prefix}${head + 1}`,
          label: `${prefix}${head + 1}`,
          shapeFormula,
          stage,
          lSymbol: 'L',
          dSymbol,
        },
        labelOptions: {
          width: 138,
          height: 34,
          fontSize: 14,
          scale: 0.054,
          fill: labelFill,
        },
      });
      const bounds = getTightCellBounds(tensor.cells, {
        marginX: 2.1,
        marginY: 2.1,
        marginZ: 2.3,
      });
      const box = addWireBox({
        label: `${prefix}${head + 1}`,
        x: bounds.centerX,
        y: bounds.centerY,
        z: bounds.centerZ,
        width: bounds.width,
        height: bounds.height,
        depth: bounds.depth,
        color,
        opacity: 0.56,
        showCenterLabel: false,
      });
      const headLabel = createOverlaySprite(`head-${head + 1}`, {
        width: 86,
        height: 26,
        fontSize: 12,
        scale: 0.046,
        fill: color,
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      headLabel.position.set(bounds.centerX - bounds.width * 0.26, bounds.centerY + bounds.height * 0.38, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(headLabel);
      const bottomFormula = createFormulaSprite(totalFormula || shapeFormula, {
        width: 220,
        height: 28,
        fontSize: 10,
        scale: 0.036,
        align: 'left',
        fill: '#dbeafe',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      bottomFormula.position.set(bounds.centerX - bounds.width * 0.12, bounds.centerY - bounds.height * 0.40, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(bottomFormula);
      const topFormula = createFormulaSprite(outputFormula || `${prefix}_i\\in\\mathbb{R}^{B\\times (L/h)\\times ${dSymbol}}`, {
        width: 220,
        height: 28,
        fontSize: 10,
        scale: 0.036,
        align: 'right',
        fill: '#fef3c7',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      topFormula.position.set(bounds.centerX + bounds.width * 0.12, bounds.centerY + bounds.height * 0.38, bounds.centerZ + bounds.depth * 0.5 + 1.5);
      interior.add(topFormula);
      items.push({
        tensor,
        box,
        headLabel,
        bottomFormula,
        topFormula,
        centerX: bounds.centerX,
        top: new THREE.Vector3(bounds.centerX, bounds.centerY + bounds.height * 0.5, bounds.centerZ),
        bottom: new THREE.Vector3(bounds.centerX, bounds.centerY - bounds.height * 0.5, bounds.centerZ),
      });
    });
    return { items };
  };

  const addExtrudedArrowPolyline = (points, color = '#38bdf8', {
    shaftWidth = 3.4,
    depth = 3.6,
    opacity = 0.78,
    showHead = true,
    headWidth = 9.5,
    headLength = 12.5,
  } = {}) => {
    if (!Array.isArray(points) || points.length < 2) return;
    const shaftMaterial = new THREE.MeshPhongMaterial({
      color,
      transparent: true,
      opacity,
      shininess: 56,
      depthWrite: false,
    });
    for (let idx = 0; idx < points.length - 1; idx += 1) {
      const start = points[idx];
      const end = points[idx + 1];
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.hypot(dx, dy);
      if (length < 0.001) continue;
      const shaft = new THREE.Mesh(
        new THREE.BoxGeometry(length, shaftWidth, depth),
        shaftMaterial.clone(),
      );
      shaft.position.set((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2);
      shaft.rotation.z = Math.atan2(dy, dx);
      interior.add(shaft);
    }
    if (!showHead) return;
    const from = points[points.length - 2];
    const to = points[points.length - 1];
    const rotationZ = Math.atan2(to.y - from.y, to.x - from.x) - Math.PI / 2;
    const head = new THREE.Mesh(
      createArrowHeadGeometry(headWidth, headLength, depth + 0.6),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: Math.min(1, opacity + 0.12),
        shininess: 62,
        depthWrite: false,
      }),
    );
    head.position.set(to.x, to.y, to.z);
    head.rotation.z = rotationZ;
    interior.add(head);
  };

  const addForkMergeArrow = (branchStarts, hub, target, color = '#38bdf8') => {
    branchStarts.forEach((start) => {
      addExtrudedArrowPolyline([
        start,
        new THREE.Vector3(start.x, hub.y, hub.z),
        hub,
      ], color, { opacity: 0.68, showHead: false });
    });
    addExtrudedArrowPolyline([hub, target], color, {
      opacity: 0.82,
      showHead: true,
      shaftWidth: 4.0,
      headWidth: 10.5,
      headLength: 13.5,
    });
  };

  const visualL = Math.max(1, Number(visualSpec.visualL || 12));
  const visualD = 6;
  const concatVisualLStep = 7.2 * 1.38;
  const inputCell = 3.4;
  const inputTargetSpan = frameWidth * 0.75;
  const inputXSpacingFactor = inputTargetSpan / (Math.max(1, visualL - 1) * inputCell);
  const inputSegments = Array.isArray(visualSpec.inputSegments) && visualSpec.inputSegments.length
    ? visualSpec.inputSegments
      .map((segment) => ({
        ...segment,
        count: Math.max(1, Number(segment?.count || 1)),
      }))
      .filter((segment) => segment && segment.label)
    : [{ label: visualSpec.inputLabel || 'X/H', count: visualL, color: '#c4b5fd' }];
  const inputLCols = visualL;
  const input = addTensorMatrix({
    label: visualSpec.inputLabel || 'X/H',
    x: 0,
    y: frameBottomY + 18,
    z: fixedZ + 4,
    lCols: inputLCols,
    dRows: visualD,
    bLayers: 3,
    color: '#c4b5fd',
    segments: inputSegments,
    segmentAxis: 'l',
    cell: inputCell,
    xSpacingFactor: inputXSpacingFactor,
    opacity: 0.48,
    batchOpacityStep: 0,
    batchShiftX: attentionBatchShift.x,
    batchShiftY: attentionBatchShift.y,
    batchShiftZ: attentionBatchShift.z,
    nodeShape: 'sphere',
    tensorInfo: {
      key: visualSpec.inputLabel || 'X_{enc}',
      label: visualSpec.inputLabel || 'X_{enc}',
      shapeFormula: visualSpec.inputShapeFormula || 'X \\in \\mathbb{R}^{B\\times L\\times d_{model}}',
      stage: 'Attention Input Tensor',
      lSymbol: visualSpec.lSymbol || 'L',
      dSymbol: visualSpec.inputDimSymbol || 'd',
    },
  });
  if (moduleEntry.id === 'encoder') {
    addBatchAxisRemapAnimation(input);
  }

  const qX = -frameWidth * 0.28;
  const kX = 0;
  const vX = frameWidth * 0.28;
  const qkvY = frameBottomY + 54;
  const qkvTensorOptions = {
    lCols: inputLCols,
    dRows: visualD,
    bLayers: 3,
    cell: 2.85,
    opacity: 0.42,
    batchOpacityStep: 0,
    batchShiftX: attentionBatchShift.x,
    batchShiftY: attentionBatchShift.y,
    batchShiftZ: attentionBatchShift.z,
    lGroupSize: 4,
    lGroupGap: 5.2,
    nodeShape: 'sphere',
    segmentAxis: 'd',
    labelOptions: {
      width: 180,
      height: 44,
      fontSize: 20,
      scale: 0.082,
      fill: '#f8fafc',
    },
  };
  const q = addTensorMatrix({ ...qkvTensorOptions, label: 'Q=XW_Q', x: qX, y: qkvY, z: fixedZ + 3, color: '#61dafb' });
  q.cells.forEach((cellInfo) => {
    cellInfo.mesh.userData.attentionTensorNode.tensorShapeFormula = 'Q \\in \\mathbb{R}^{B\\times L\\times d}';
    cellInfo.mesh.userData.attentionTensorNode.tensorStage = 'Q Projection Tensor';
    cellInfo.mesh.userData.attentionTensorNode.dSymbol = 'd';
  });
  const k = addTensorMatrix({ ...qkvTensorOptions, label: 'K=XW_K', x: kX, y: qkvY, z: fixedZ + 3, color: '#7ee787' });
  k.cells.forEach((cellInfo) => {
    cellInfo.mesh.userData.attentionTensorNode.tensorShapeFormula = 'K \\in \\mathbb{R}^{B\\times L\\times d}';
    cellInfo.mesh.userData.attentionTensorNode.tensorStage = 'K Projection Tensor';
    cellInfo.mesh.userData.attentionTensorNode.dSymbol = 'd';
  });
  const v = addTensorMatrix({ ...qkvTensorOptions, label: 'V=XW_V', x: vX, y: qkvY, z: fixedZ + 3, color: '#a78bfa' });
  v.cells.forEach((cellInfo) => {
    cellInfo.mesh.userData.attentionTensorNode.tensorShapeFormula = 'V \\in \\mathbb{R}^{B\\times L\\times d}';
    cellInfo.mesh.userData.attentionTensorNode.tensorStage = 'V Projection Tensor';
    cellInfo.mesh.userData.attentionTensorNode.dSymbol = 'd';
  });
  const qHeadFrames = addInlineHeadSplitFrames({
    tensorLabel: 'Q',
    sourceTensor: q,
    color: '#61dafb',
    totalL: visualL,
    heads: 3,
  });
  const kHeadFrames = addInlineHeadSplitFrames({
    tensorLabel: 'K',
    sourceTensor: k,
    color: '#7ee787',
    totalL: visualL,
    heads: 3,
  });
  const vHeadFrames = addInlineHeadSplitFrames({
    tensorLabel: 'V',
    sourceTensor: v,
    color: '#a78bfa',
    totalL: visualL,
    heads: 3,
  });

  const perHeadL = Math.ceil(visualL / 3);
  const outputY = frameTopY - 22;
  const stageGap = (outputY - qkvY) / 4;
  const sRowY = qkvY + stageGap;
  const aRowY = qkvY + stageGap * 2;
  const oRowY = qkvY + stageGap * 3;
  const concatY = oRowY + (outputY - oRowY) * 0.5;
  const sHeadCenters = qHeadFrames.items.map((item, head) => (
    (item.centerX + kHeadFrames.items[head].centerX) / 2
  ));
  const sHeads = addHeadTensorRow({
    prefix: 'S_e',
    centers: sHeadCenters,
    y: sRowY,
    z: fixedZ + 4,
    lCols: perHeadL,
    dRows: perHeadL,
    bLayers: 3,
    color: '#fbbf24',
    shapeFormula: 'S_e \\in \\mathbb{R}^{B\\times L\\times L}',
    stage: 'Score Tensor',
    totalFormula: 'S\\in\\mathbb{R}^{B\\times L\\times L}',
    outputFormula: 'S_i\\in\\mathbb{R}^{B\\times (L/h)\\times (L/h)}',
    dSymbol: 'L',
    labelFill: '#fde68a',
  });
  const aHeads = addHeadTensorRow({
    prefix: 'A_e',
    centers: sHeadCenters,
    y: aRowY,
    z: fixedZ + 4,
    lCols: perHeadL,
    dRows: perHeadL,
    bLayers: 3,
    color: '#fb7185',
    shapeFormula: 'A_e \\in \\mathbb{R}^{B\\times L\\times L}',
    stage: 'Attention Weight Tensor',
    totalFormula: 'A\\in\\mathbb{R}^{B\\times L\\times L}',
    outputFormula: 'A_i\\in\\mathbb{R}^{B\\times (L/h)\\times (L/h)}',
    dSymbol: 'L',
    labelFill: '#fecdd3',
  });
  const applyFullRelationVisual = (tensorRow, {
    relationColor = null,
    relationOpacity = 0.58,
    diagonalColor = '#fef3c7',
    diagonalOpacity = 0.90,
    activeEmissive = 0.18,
  } = {}) => {
    tensorRow.items.forEach((item) => {
      item.tensor.cells.forEach((cellInfo) => {
        const material = cellInfo.mesh.material;
        if (!material) return;
        if (cellInfo.d === cellInfo.l) {
          const diagColor = diagonalColor || cellInfo.color || '#fef3c7';
          material.color.set(diagColor);
          material.emissive.set(diagColor);
          material.opacity = diagonalOpacity;
          material.emissiveIntensity = activeEmissive;
          return;
        }
        const liveColor = relationColor || cellInfo.color || '#93c5fd';
        material.color.set(liveColor);
        material.emissive.set(liveColor);
        material.opacity = relationOpacity;
        material.emissiveIntensity = activeEmissive * 0.72;
      });
      if (item.box?.edges?.material) {
        item.box.edges.material.opacity = Math.max(item.box.edges.material.opacity || 0, 0.56);
      }
    });
  };
  if (visualSpec.relationMatrixEmphasis) {
    applyFullRelationVisual(sHeads, { relationOpacity: 0.54, diagonalColor: '#fde68a', diagonalOpacity: 0.92 });
    applyFullRelationVisual(aHeads, { relationOpacity: 0.62, diagonalColor: '#fda4af', diagonalOpacity: 0.88 });
  }
  const oHeadCenters = sHeadCenters.map((center, head) => (
    (center + vHeadFrames.items[head].centerX) / 2
  ));
  const oHeads = addHeadTensorRow({
    prefix: 'O_e',
    centers: oHeadCenters,
    y: oRowY,
    z: fixedZ + 4,
    lCols: perHeadL,
    dRows: visualD,
    bLayers: 3,
    color: '#93c5fd',
    shapeFormula: 'O_e \\in \\mathbb{R}^{B\\times L\\times d}',
    stage: 'Head Output Tensor',
    totalFormula: 'O\\in\\mathbb{R}^{B\\times L\\times d}',
    outputFormula: 'O_i\\in\\mathbb{R}^{B\\times (L/h)\\times d}',
    dSymbol: 'd',
    labelFill: '#bfdbfe',
  });
  const outputXSpacingFactor = concatVisualLStep / 2.85;
  const output = addTensorMatrix({
    label: visualSpec.outputLabel || 'H_out',
    x: 0,
    y: outputY,
    z: fixedZ + 3,
    lCols: inputLCols,
    dRows: visualD,
    bLayers: 3,
    color: '#38bdf8',
    cell: 2.85,
    opacity: 0.46,
    batchOpacityStep: 0,
    batchShiftX: attentionBatchShift.x,
    batchShiftY: attentionBatchShift.y,
    batchShiftZ: attentionBatchShift.z,
    xSpacingFactor: outputXSpacingFactor,
    nodeShape: 'sphere',
    segmentAxis: 'd',
    labelOptions: {
      width: 170,
      height: 42,
      fontSize: 18,
      scale: 0.074,
      fill: '#bae6fd',
    },
    tensorInfo: {
      key: visualSpec.outputLabel || 'H_{out}',
      label: visualSpec.outputLabel || 'H_{out}',
      shapeFormula: visualSpec.headOutputFormula || 'H \\in \\mathbb{R}^{B\\times L\\times d}',
      stage: 'Attention Output Tensor',
      lSymbol: visualSpec.lSymbol || 'L',
      dSymbol: visualSpec.outputDimSymbol || 'd',
    },
  });
  const outputBounds = getTightCellBounds(output.cells, {
    marginX: 2.2,
    marginY: 2.2,
    marginZ: 2.4,
  });
  const outputFrame = addWireBox({
    label: '',
    x: outputBounds.centerX,
    y: outputBounds.centerY,
    z: outputBounds.centerZ,
    width: outputBounds.width,
    height: outputBounds.height,
    depth: outputBounds.depth,
    color: '#38bdf8',
    opacity: 0.62,
    showCenterLabel: false,
  });

  addSourceColoredCellConnections(input.cells, q.cells, 0.006);
  addSourceColoredCellConnections(input.cells, k.cells, 0.006);
  addSourceColoredCellConnections(input.cells, v.cells, 0.006);
  qHeadFrames.items.forEach((frame, head) => {
    const qSlice = q.cells.filter((entry) => entry.l >= frame.lStart && entry.l <= frame.lEnd);
    const kSlice = k.cells.filter((entry) => entry.l >= frame.lStart && entry.l <= frame.lEnd);
    addSourceColoredCellConnections(qSlice, sHeads.items[head].tensor.cells, 0.012);
    addSourceColoredCellConnections(kSlice, sHeads.items[head].tensor.cells, 0.012);
    interior.add(createLine([frame.top, sHeads.items[head].box.bottom], '#61dafb', 0.48));
    interior.add(createLine([kHeadFrames.items[head].top, sHeads.items[head].box.bottom], '#7ee787', 0.48));
  });
  sHeads.items.forEach((item, head) => {
    addSourceColoredCellConnections(item.tensor.cells, aHeads.items[head].tensor.cells, 0.010);
    interior.add(createLine([item.box.top, aHeads.items[head].box.bottom], '#fb7185', 0.56));
  });
  aHeads.items.forEach((item, head) => {
    const vSlice = v.cells.filter((entry) => entry.l >= vHeadFrames.items[head].lStart && entry.l <= vHeadFrames.items[head].lEnd);
    addSourceColoredCellConnections(item.tensor.cells, oHeads.items[head].tensor.cells, 0.01);
    addSourceColoredCellConnections(vSlice, oHeads.items[head].tensor.cells, 0.01);
    interior.add(createLine([item.box.top, oHeads.items[head].box.bottom], '#93c5fd', 0.48));
    interior.add(createLine([vHeadFrames.items[head].top, oHeads.items[head].box.bottom], '#a78bfa', 0.48));
  });
	  addFormulaAt('X,Q,K,V,H_{out} \\in \\mathbb{R}^{B\\times L\\times d}', 0, frameBottomY + 37, {
	    width: 650,
	    height: 38,
	    fontSize: 10,
	    fill: '#fef9c3',
	    scale: 0.04,
	  });
	  addFormulaAt('visual axes: x=L,\\ y=B,\\ z=d;\\ L=12,\\ d=6', frameWidth * 0.30, frameBottomY + 17, {
	    width: 360,
	    height: 34,
	    fontSize: 10,
	    fill: '#bfdbfe',
	    scale: 0.04,
	  });
  addFormulaAt(
    visualSpec.inputShapeFormula || 'X \\in \\mathbb{R}^{B\\times L\\times d_{model}}',
    0,
    frameBottomY + 4,
    { width: 430, fill: '#c4b5fd', fontSize: 12 },
  );
  addFormulaAt(
    visualSpec.projectionFormula || 'Q=Proj_Q(X),\\ K=Proj_K(X),\\ V=Proj_V(X)',
    0,
    frameBottomY + 30,
    { width: 680, fill: '#fde68a', fontSize: 11, scale: 0.046 },
  );
  addFormulaAt(
    visualSpec.qkvShapeFormula || 'Q,K,V \\in \\mathbb{R}^{B\\times L\\times d} \\rightarrow \\{Q_i,K_i,V_i\\}_{i=1}^{h},\\ Q_i \\in \\mathbb{R}^{B\\times (L/h)\\times d}',
    0,
    frameBottomY + 96,
    { width: 720, fill: '#bfdbfe', fontSize: 11, scale: 0.046 },
  );
  if (visualSpec.relationMatrixEmphasis) {
    addFormulaAt(
      'Full set attention: S_i[t,j]\\ remains\\ valid\\ for\\ all\\ candidate\\ slots',
      0,
      frameBottomY + 126,
      { width: 680, fill: '#bbf7d0', fontSize: 11, scale: 0.046 },
    );
  }

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    activeObjects.forEach((obj, index) => {
      const sPulse = 1 + Math.sin(elapsed * 1.9 + index * 0.37) * 0.035;
      obj.scale.setScalar(sPulse);
    });
    const framePulse = 1 + Math.sin(elapsed * 2.6) * 0.018;
    outputFrame.mesh.scale.set(framePulse, framePulse, framePulse);
    outputFrame.edges.scale.set(framePulse, framePulse, framePulse);
    outputFrame.edges.material.opacity = 0.46 + (Math.sin(elapsed * 2.6) * 0.5 + 0.5) * 0.28;
    batchRemapObjects.forEach((entry) => {
      const cycle = (elapsed * 0.16) % 1;
      const raw = (cycle - entry.phase + 1) % 1;
      const moveWindow = 0.18;
      const settleWindow = 0.08;
      const fadeWindow = 0.10;
      let eased = 0;
      let alpha = entry.baseOpacity * 0.08;

      if (raw < moveWindow) {
        const t = raw / moveWindow;
        eased = 1 - ((1 - t) ** 3);
        alpha = entry.baseOpacity * (0.96 - eased * 0.18);
      } else if (raw < moveWindow + settleWindow) {
        eased = 1;
        alpha = entry.baseOpacity * 0.78;
      } else if (raw < moveWindow + settleWindow + fadeWindow) {
        const t = (raw - moveWindow - settleWindow) / fadeWindow;
        eased = 1;
        alpha = entry.baseOpacity * (0.78 * (1 - t));
      }

      entry.mesh.position.lerpVectors(entry.start, entry.end, eased);
      entry.mesh.material.opacity = alpha;
      entry.mesh.scale.setScalar(0.74 + Math.sin(elapsed * 4.2 + entry.layer * 1.35) * 0.04);
    });
    headSplitObjects.forEach((entry) => {
      const t = 0.5 + Math.sin(elapsed * 0.9 + entry.phase * 8) * 0.5;
      const eased = 1 - ((1 - t) ** 3);
      const x = THREE.MathUtils.lerp(entry.startX, entry.targetX, eased);
      const scaleX = THREE.MathUtils.lerp(entry.startScaleX || 1, entry.targetScaleX || 1, eased);
      entry.mesh.position.x = x;
      entry.edges.position.x = x;
      entry.mesh.scale.x = scaleX;
      entry.edges.scale.x = scaleX;
      if (entry.labelSprite) {
        entry.labelSprite.position.x = x;
      }
      if (entry.titleSprite) {
        entry.titleSprite.position.x = x - (entry.boxWidth * scaleX) * 0.34;
        entry.titleSprite.position.y = entry.mesh.position.y + entry.boxHeight * 0.38;
        entry.titleSprite.position.z = entry.fixedZ + entry.boxDepth * 0.5 + 1.5;
      }
      if (entry.inputSprite) {
        entry.inputSprite.position.x = x - (entry.boxWidth * scaleX) * 0.18;
        entry.inputSprite.position.y = entry.mesh.position.y - entry.boxHeight * 0.40;
        entry.inputSprite.position.z = entry.fixedZ + entry.boxDepth * 0.5 + 1.5;
      }
      if (entry.outputSprite) {
        entry.outputSprite.position.x = x + (entry.boxWidth * scaleX) * 0.18;
        entry.outputSprite.position.y = entry.mesh.position.y + entry.boxHeight * 0.38;
        entry.outputSprite.position.z = entry.fixedZ + entry.boxDepth * 0.5 + 1.5;
      }
      if (entry.interiorGroup) {
        entry.interiorGroup.position.x = x;
        entry.interiorGroup.position.z = entry.fixedZ;
      }
      if (entry.innerCellTargets?.length) {
        const layerOpacityScale = Math.max(0.42, 1 - entry.headIndex * 0.22);
        entry.innerCellTargets.forEach((cellInfo) => {
          cellInfo.mesh.material.opacity = (0.14 + eased * 0.20) * layerOpacityScale;
          cellInfo.mesh.material.emissiveIntensity = (0.04 + eased * 0.08) * layerOpacityScale;
        });
      }
      entry.edges.material.opacity = entry.baseOpacity * (0.62 + eased * 0.38);
    });
  });
  moduleEntry.interior = interior;
}

function addHeadsVisual(group, width, height) {
  const values = [0.22, 0.36, 0.18, 0.24];
  const colors = ['#61dafb', '#7ee787', '#fbbf24', '#fb7185'];
  values.forEach((value, idx) => {
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(16, value * height * 0.6 + 22, 12),
      new THREE.MeshPhongMaterial({ color: colors[idx], transparent: true, opacity: 0.74 }),
    );
    bar.position.set(-width * 0.24 + idx * width * 0.16, -18 + bar.geometry.parameters.height / 2, 8);
    group.add(bar);
  });
}

function addAnalysisVisual(group, width, height) {
  const values = [0.28, 0.54, 0.38, 0.62, 0.48];
  values.forEach((value, idx) => {
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(14, value * height * 0.6 + 18, 10),
      new THREE.MeshPhongMaterial({ color: lerpColorHex('#61dafb', '#7ee787', idx / values.length), transparent: true, opacity: 0.7 }),
    );
    bar.position.set(-width * 0.26 + idx * 22, -18 + bar.geometry.parameters.height / 2, 8);
    group.add(bar);
  });
  const line = createLine([
    new THREE.Vector3(-width * 0.3, -height * 0.18, 14),
    new THREE.Vector3(-width * 0.15, -2, 14),
    new THREE.Vector3(0, -height * 0.04, 14),
    new THREE.Vector3(width * 0.16, height * 0.12, 14),
    new THREE.Vector3(width * 0.3, height * 0.18, 14),
  ], '#fbbf24', 0.85);
  group.add(line);
}

function addWeightVisual(group, width, height, color) {
  for (let i = 0; i < 3; i += 1) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(width * 0.14 + i * 6, 4, 12, 40),
      new THREE.MeshPhongMaterial({ color: lerpColorHex(color, '#ffffff', i * 0.12), transparent: true, opacity: 0.8 }),
    );
    ring.rotation.x = Math.PI / 2.6;
    ring.position.z = i * 5;
    group.add(ring);
  }
}

function drawRoundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function createPanelTexture(text, {
  width = 820,
  height = 240,
  fill = '#f7f7f2',
  border = '#111111',
  textColor = '#111111',
  fontSize = 56,
  lineHeight = 1.18,
  borderWidth = 10,
  radius = 26,
  shadow = false,
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  if (shadow) {
    ctx.fillStyle = 'rgba(0,0,0,0.08)';
    drawRoundRect(ctx, 18, 18, width - 20, height - 20, radius);
    ctx.fill();
  }
  ctx.fillStyle = fill;
  ctx.strokeStyle = border;
  ctx.lineWidth = borderWidth;
  drawRoundRect(ctx, 8, 8, width - 16, height - 16, radius);
  ctx.fill();
  ctx.stroke();

  const lines = String(text || '').split('\n');
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `700 ${fontSize}px "Songti SC", "STSong", "Noto Serif SC", serif`;
  const actualLineHeight = fontSize * lineHeight;
  const totalHeight = actualLineHeight * lines.length;
  const startY = height / 2 - totalHeight / 2 + actualLineHeight / 2;
  lines.forEach((line, index) => {
    ctx.fillText(line, width / 2, startY + index * actualLineHeight);
  });

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createModuleSurfaceTexture(spec, {
  width = 840,
  height = 360,
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, width, height);
  const padX = Math.max(18, width * 0.03);
  const padY = Math.max(8, height * 0.03);
  const surfaceTitle = MODULE_TITLE_TEXT[spec.id] || spec.surfaceLabel || spec.label || '';
  const label = String(surfaceTitle).split('\n');
  const fontSize = Math.max(14, Math.min(26, height * 0.16));
  const lineHeight = fontSize * 1.06;

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = spec.textColor || '#f8fafc';
  ctx.font = `700 ${fontSize}px "Avenir Next", "PingFang SC", Arial, sans-serif`;
  label.forEach((line, index) => {
    ctx.fillText(line, padX, padY + index * lineHeight);
  });

  const formulaSpec = MODULE_FORMULAS[spec.id];
  const fitFormulaSize = (formula, preferred, maxWidth) => {
    let size = preferred;
    while (size > 10 && measureFormulaWidth(ctx, formula, size) > maxWidth) size -= 1;
    return size;
  };
  if (formulaSpec?.output) {
    const outputFont = fitFormulaSize(
      formulaSpec.output,
      Math.max(12, Math.min(18, height * 0.11)),
      width * 0.42,
    );
    drawFormulaLine(
      ctx,
      formulaSpec.output,
      width - padX,
      padY + outputFont,
      {
        fontSize: outputFont,
        color: '#fef3c7',
        align: 'right',
      },
    );
  }
  if (formulaSpec?.input) {
    const inputFont = fitFormulaSize(
      formulaSpec.input,
      Math.max(11, Math.min(17, height * 0.105)),
      width * 0.44,
    );
    drawFormulaLine(
      ctx,
      formulaSpec.input,
      padX,
      height - padY - 6,
      {
        fontSize: inputFont,
        color: '#dbeafe',
        align: 'left',
      },
    );
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createTexturedThinBox({
  width,
  height,
  depth = 2,
  texture,
  sideColor = '#f8fafc',
  sideOpacity = 1,
  shininess = 24,
}) {
  const sideMaterial = new THREE.MeshPhongMaterial({
    color: sideColor,
    transparent: false,
    opacity: sideOpacity,
    shininess,
  });
  const faceMaterial = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: false,
  });
  const materials = [
    sideMaterial.clone(),
    sideMaterial.clone(),
    sideMaterial.clone(),
    sideMaterial.clone(),
    faceMaterial,
    faceMaterial.clone(),
  ];
  return new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), materials);
}

function createModuleTitleTag(text, {
  width = 180,
  height = 42,
  depth = 8,
  fill = '#f7f7f2',
  border = '#111111',
  textColor = '#111111',
  fontSize = 30,
  opacity = 0.88,
} = {}) {
  const group = new THREE.Group();
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const faceTexture = createPanelTexture(text, {
    width: Math.max(520, Math.round(width * 3.4)),
    height: Math.max(140, Math.round(height * 3.3)),
    fill,
    border: 'rgba(0,0,0,0)',
    textColor,
    fontSize,
    borderWidth: 0,
    radius: 18,
    shadow: false,
  });

  const box = new THREE.Mesh(
    geometry,
    [
      new THREE.MeshPhongMaterial({ color: fill, transparent: false, opacity, shininess: 28 }),
      new THREE.MeshPhongMaterial({ color: fill, transparent: false, opacity, shininess: 28 }),
      new THREE.MeshPhongMaterial({ color: fill, transparent: false, opacity, shininess: 28 }),
      new THREE.MeshPhongMaterial({ color: fill, transparent: false, opacity, shininess: 28 }),
      new THREE.MeshBasicMaterial({ map: faceTexture, transparent: false }),
      new THREE.MeshBasicMaterial({ map: faceTexture.clone(), transparent: false }),
    ],
  );
  box.renderOrder = 40;
  group.add(box);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: border,
      transparent: false,
      opacity: 1,
    }),
  );
  edges.position.z = 0.5;
  edges.renderOrder = 41;
  group.add(edges);

  return group;
}

const MODULE_TITLE_TEXT = {
  x_rp: 'X_rp',
  x_spCtx: 'X_hist',
  x_env: 'X_env',
  x_cand: 'X_cand',
  inputEmbedding: 'Input Embedding',
  concatPanel: 'Concat',
  encoder: 'Multi-Head Attention',
  encoderAddNormBottom: 'Add & Norm',
  encoderFeedForward: 'Feed Forward',
  encoderAddNormTop: 'Add & Norm',
  outputEmbedding: 'Embedding&Concat',
  questionSetAttention: 'Question-Set Attention',
  decoderAddNormSelf: 'Add & Norm',
  decoderCrossAttention: 'Cross Attention',
  decoderAddNormCross: 'Add & Norm',
  decoder: 'Feed Forward',
  decoderAddNormTop: 'Add & Norm',
  maskHead: 'Mask Head',
  valueHead: 'Value Head',
  scoreHead: 'Score Head',
  countHead: 'Count Head',
  slotSelectHead: 'Slot Select Head',
  outputBlock: 'Question Block',
  weights: 'Weights',
  analysis: 'Analysis',
};

const MODULE_INTERIOR_SPECS = {
  x_rp: {
    variables: ['gender', 'age_grp', 'edu', 'hh_size'],
    tint: '#61dafb',
  },
  x_spCtx: {
    variables: ['rho_v', 'mean_v', 'std_v', 'spread'],
    tint: '#fbbf24',
  },
  x_env: {
    variables: ['progress', 'zone_gap', 'd_error', 'stability'],
    tint: '#7ee787',
  },
  x_cand: {
    variables: ['alt_id', 'var_id', 'lower', 'upper'],
    tint: '#a78bfa',
  },
  inputEmbedding: {
    tint: '#f59e0b',
    networks: [
      {
        key: 'rp',
        sourceId: 'x_rp',
        title: 'E_rp',
        inputCount: 'd_{rp}',
        hiddenCount: 'h_{rp}',
        outputCount: 'd_{model}',
        sequenceLength: '1',
        inputDim: 'd_{rp}',
        hiddenDim: 'h_{rp}',
        outputDim: 'd_{model}',
        color: '#61dafb',
        displayCounts: [9, 7],
      },
      {
        key: 'spCtx',
        sourceId: 'x_spCtx',
        title: 'E_hist',
        inputCount: 'd_{hist}',
        hiddenCount: 'h_{spCtx}',
        outputCount: 'd_{model}',
        sequenceLength: 'L_{hist}',
        inputDim: 'd_{hist}',
        hiddenDim: 'h_{ctx}',
        outputDim: 'd_{model}',
        color: '#fbbf24',
        displayCounts: [7, 8],
      },
      {
        key: 'env',
        sourceId: 'x_env',
        title: 'E_env',
        inputCount: 'd_{env}',
        hiddenCount: 'h_{env}',
        outputCount: 'd_{model}',
        sequenceLength: '1',
        inputDim: 'd_{env}',
        hiddenDim: 'h_{env}',
        outputDim: 'd_{model}',
        color: '#7ee787',
        displayCounts: [6, 7],
      },
      {
        key: 'cand',
        sourceId: 'x_cand',
        title: 'E_cand',
        inputCount: 'd_{cand}',
        hiddenCount: 'h_{cand}',
        outputCount: 'd_{model}',
        sequenceLength: 'L_{cand}',
        inputDim: 'd_{cand}',
        hiddenDim: 'h_{cand}',
        outputDim: 'd_{model}',
        color: '#a78bfa',
        displayCounts: [10, 8],
      },
    ],
  },
  concatPanel: {
    tint: '#93c5fd',
    concatBlocks: [
      {
        key: 'E_{rp}',
        tokenFormula: 'E_{rp}',
        shapeFormula: 'E_{rp} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}',
        sourceFormula: 'X_{rp} \\rightarrow E_{rp}',
        tokenCountFormula: '1',
        displayTokens: 1,
        color: '#61dafb',
      },
      {
        key: 'E_{hist}',
        tokenFormula: 'E_{hist}',
        shapeFormula: 'E_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{model}}',
        sourceFormula: 'X_{hist} \\rightarrow E_{hist}',
        tokenCountFormula: 'L_{hist}',
        displayTokens: 5,
        color: '#fbbf24',
      },
      {
        key: 'E_{env}',
        tokenFormula: 'E_{env}',
        shapeFormula: 'E_{env} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}',
        sourceFormula: 'X_{env} \\rightarrow E_{env}',
        tokenCountFormula: '1',
        displayTokens: 1,
        color: '#7ee787',
      },
      {
        key: 'E_{cand}',
        tokenFormula: 'E_{cand}',
        shapeFormula: 'E_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{model}}',
        sourceFormula: 'X_{cand} \\rightarrow E_{cand}',
        tokenCountFormula: 'L_{cand}',
        displayTokens: 5,
        color: '#a78bfa',
      },
      {
        key: 'X_{enc}',
        tokenFormula: 'X_{enc}',
        shapeFormula: 'X_{enc}^{(viz)} \\in \\mathbb{R}^{B\\times (1+L_{hist}+1+L_{cand})\\times 6}',
        sourceFormula: 'concat(E_{rp},E_{hist},E_{env},E_{cand})',
        tokenCountFormula: 'L=1+5+1+5=12,\\ d=6',
        isFinal: true,
        color: '#93c5fd',
      },
    ],
  },
  encoder: {
    kind: 'multiheadAttention',
    tint: '#fbbf24',
    headLabel: 'Head 1 / Self-Attention',
    inputLabel: 'X_{enc}',
    inputSegments: [
      { label: 'E_rp', count: 1, color: '#61dafb' },
      { label: 'E_hist', count: 5, color: '#fbbf24' },
      { label: 'E_env', count: 1, color: '#7ee787' },
      { label: 'E_cand', count: 5, color: '#a78bfa' },
    ],
    inputShapeFormula: 'X_{enc} \\in \\mathbb{R}^{B\\times L\\times d}',
    projectionFormula: 'Q=Proj_Q(X_{enc}),\\ K=Proj_K(X_{enc}),\\ V=Proj_V(X_{enc})',
    qkvShapeFormula: 'Q,K,V \\in \\mathbb{R}^{B\\times L\\times d} \\rightarrow \\{Q_i,K_i,V_i\\}_{i=1}^{h},\\ Q_i \\in \\mathbb{R}^{B\\times (L/h)\\times d}',
    scoreShapeFormula: 'S_i,A_i \\in \\mathbb{R}^{B\\times (L/h)\\times (L/h)}',
    headOutputFormula: 'H_{enc} \\in \\mathbb{R}^{B\\times L\\times d}',
    formula: 'S_i=Q_iK_i^T/\\sqrt{d}, A_i=softmax(S_i), O_i=A_iV_i',
    shapeFormula: 'Q_i,K_i,V_i \\in \\mathbb{R}^{B\\times (L/h)\\times d}, O_i \\in \\mathbb{R}^{B\\times (L/h)\\times d}',
  },
  encoderAddNormBottom: {
    kind: 'addNorm',
    tint: '#fde68a',
    concatLabel: 'X_{enc}',
    attentionLabel: 'H_{out}',
    outputLabel: 'H^{norm}',
    concatColor: '#93c5fd',
    attentionColor: '#38bdf8',
    outputColor: '#fef3c7',
  },
  encoderFeedForward: {
    kind: 'feedForward',
    tint: '#93c5fd',
    inputLabel: 'H^{norm}',
    outputLabel: 'FFN(H)',
    dimFormula: 'B\\times L\\times d',
    color: '#93c5fd',
    inputColor: '#fef3c7',
    outputColor: '#93c5fd',
  },
  encoderAddNormTop: {
    kind: 'addNorm',
    tint: '#fde68a',
    concatLabel: 'H^{norm}',
    attentionLabel: 'FFN(H)',
    outputLabel: 'H_{enc}',
    concatColor: '#fef3c7',
    attentionColor: '#93c5fd',
    outputColor: '#fef3c7',
    concatNodeShape: 'sphere',
    attentionNodeShape: 'sphere',
    addFormulaText: 'Z^{ff}_{b,l,d}=H^{norm}_{b,l,d}+FFN(H)_{b,l,d}',
    normFormulaText: 'H_{enc}=LayerNorm(Z^{ff})',
  },
  outputEmbedding: {
    kind: 'outputEmbedding',
    tint: '#f59e0b',
    inputLabel: 'Q_{slot}^{0}',
    outputLabel: 'Q_{slot}',
    dimFormula: 'B\\times T_{max}\\times d_{model}',
    networks: [
      {
        key: 'rp',
        sourceId: 'x_rp',
        title: 'E_rp',
        inputCount: 'd_{rp}',
        outputCount: 'd_{model}',
        sequenceLength: '1',
        color: '#61dafb',
        batchCount: 3,
        inputDCount: 6,
        outputDCount: 5,
      },
      {
        key: 'spCtx',
        sourceId: 'x_spCtx',
        title: 'E_hist',
        inputCount: 'd_{hist}',
        outputCount: 'd_{model}',
        sequenceLength: 'L_{hist}',
        color: '#fbbf24',
        batchCount: 3,
        sequenceCount: 4,
        inputDCount: 4,
        outputDCount: 5,
      },
      {
        key: 'env',
        sourceId: 'x_env',
        title: 'E_env',
        inputCount: 'd_{env}',
        outputCount: 'd_{model}',
        sequenceLength: '1',
        color: '#7ee787',
        batchCount: 3,
        inputDCount: 4,
        outputDCount: 5,
      },
      {
        key: 'cand',
        sourceId: 'x_cand',
        title: 'E_cand',
        inputCount: 'd_{cand}',
        outputCount: 'd_{model}',
        sequenceLength: 'L_{cand}',
        color: '#a78bfa',
        batchCount: 3,
        sequenceCount: 4,
        inputDCount: 4,
        outputDCount: 5,
      },
    ],
    concatBlocks: [
      { key: 'E_{rp}', displayTokens: 1, color: '#61dafb' },
      { key: 'E_{hist}', displayTokens: 5, color: '#fbbf24' },
      { key: 'E_{env}', displayTokens: 1, color: '#7ee787' },
      { key: 'E_{cand}', displayTokens: 5, color: '#a78bfa' },
    ],
    concatOutputFormula: 'Q_{slot}^{(viz)} \\in \\mathbb{R}^{3\\times 12\\times 4}',
    tokenColor: '#fda4af',
    tensorColor: '#f59e0b',
    tokenCount: 10,
    dRows: 5,
    bLayers: 3,
    tokenLabels: ['q_1', 'q_2', 'q_3', '...', 'q_{Tmax}'],
  },
  questionSetAttention: {
    kind: 'multiheadAttention',
    tint: '#fbbf24',
    relationMatrixEmphasis: false,
    headLabel: 'Head 1 / Question-Set Self-Attention',
    inputLabel: 'Q_{slot}',
    inputSegments: [
      { label: 'Q_{slot}', count: 6, color: '#f59e0b' },
    ],
    visualL: 6,
    inputShapeFormula: 'Q_{slot} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    projectionFormula: 'Q=Q_{slot}W_Q,K=Q_{slot}W_K,V=Q_{slot}W_V; W_* \\in \\mathbb{R}^{d_{model}\\times (h d_h)}',
    qkvShapeFormula: 'Q,K,V \\in \\mathbb{R}^{B\\times T_{max}\\times (h d_h)} \\rightarrow \\mathbb{R}^{B\\times h\\times T_{max}\\times d_h}',
    scoreShapeFormula: 'S_i,A_i \\in \\mathbb{R}^{B\\times T_{max}\\times T_{max}}',
    headOutputFormula: 'H_{self}=Concat(O_1,\\ldots,O_h)W_O \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    outputLabel: 'H_{slot}',
    formula: 'S_i=Q_iK_i^T/\\sqrt{d_h}, A_i=softmax(S_i), O_i=A_iV_i',
    shapeFormula: 'Q_i,K_i,V_i \\in \\mathbb{R}^{B\\times T_{max}\\times d_h}, S_i \\in \\mathbb{R}^{B\\times T_{max}\\times T_{max}}',
    lSymbol: 'T_{max}',
    inputDimSymbol: 'd_{model}',
    outputDimSymbol: 'd_{model}',
  },
  decoderAddNormSelf: {
    kind: 'addNorm',
    tint: '#fde68a',
    concatLabel: 'Q_{slot}',
    attentionLabel: 'H_{slot}',
    outputLabel: 'H^{norm}_{slot}',
    concatColor: '#f59e0b',
    attentionColor: '#38bdf8',
    outputColor: '#fef3c7',
    concatNodeShape: 'sphere',
    attentionNodeShape: 'sphere',
    addFormulaText: 'Z^{slot}_{b,t,d}=Q_{slot,b,t,d}+H_{slot,b,t,d}',
    normFormulaText: 'H^{norm}_{slot}=LayerNorm(Z^{slot})',
    outputFormulaText: 'B\\times T_{max}\\times d_{model}',
    lCols: 6,
    dRows: 6,
    bLayers: 3,
  },
  decoderCrossAttention: {
    kind: 'multiheadAttention',
    tint: '#fbbf24',
    headLabel: 'Head 1 / Cross-Attention',
    inputLabel: 'H_{slot}/H_{enc}',
    inputSegments: [
      { label: 'H^{norm}_{slot}', count: 6, color: '#fef3c7' },
    ],
    visualL: 6,
    inputShapeFormula: 'Q\\leftarrow H_{dec}, K,V\\leftarrow H_{enc}',
    projectionFormula: 'Q=H_{dec}W_Q,K=H_{enc}W_K,V=H_{enc}W_V; W_* \\in \\mathbb{R}^{d_{model}\\times (h d_h)}',
    qkvShapeFormula: 'Q \\in \\mathbb{R}^{B\\times h\\times T_{max}\\times d_h}, K,V \\in \\mathbb{R}^{B\\times h\\times L_{enc}\\times d_h}',
    scoreShapeFormula: 'S_i,A_i \\in \\mathbb{R}^{B\\times T_{max}\\times L_{enc}}',
    headOutputFormula: 'H_{cross}=Concat(O_1,\\ldots,O_h)W_O \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    outputLabel: 'O_{cross}',
    formula: 'Q_i=H_{slot}W_Q^i, K_i=H_{enc}W_K^i, V_i=H_{enc}W_V^i, O_i=softmax(S_i)V_i',
    shapeFormula: 'S_i \\in \\mathbb{R}^{B\\times T_{max}\\times L_{enc}}, O_i \\in \\mathbb{R}^{B\\times T_{max}\\times d_h}',
    lSymbol: 'T_{max}',
    inputDimSymbol: 'd_{model}',
    outputDimSymbol: 'd_{model}',
  },
  decoderAddNormCross: {
    kind: 'addNorm',
    tint: '#fde68a',
    concatLabel: 'H^{norm}_{slot}',
    attentionLabel: 'O_{cross}',
    outputLabel: 'H^{norm}_{cross}',
    concatColor: '#fef3c7',
    attentionColor: '#38bdf8',
    outputColor: '#fef3c7',
    concatNodeShape: 'sphere',
    attentionNodeShape: 'sphere',
    addFormulaText: 'Z^{cross}_{b,t,d}=H^{norm}_{slot,b,t,d}+O_{cross,b,t,d}',
    normFormulaText: 'H^{norm}_{cross}=LayerNorm(Z^{cross})',
    outputFormulaText: 'B\\times T_{max}\\times d_{model}',
    lCols: 6,
    dRows: 6,
    bLayers: 3,
  },
  decoder: {
    kind: 'feedForward',
    tint: '#93c5fd',
    inputLabel: 'H^{norm}_{cross}',
    outputLabel: 'H_{dec}',
    dimFormula: 'B\\times T_{max}\\times d_{model}',
    color: '#93c5fd',
    inputColor: '#fef3c7',
    outputColor: '#93c5fd',
    lCols: 6,
    dRows: 6,
    bLayers: 3,
  },
  decoderAddNormTop: {
    kind: 'addNorm',
    tint: '#fde68a',
    concatLabel: 'H^{norm}_{cross}',
    attentionLabel: 'H_{dec}',
    outputLabel: 'H^{out}_{dec}',
    concatColor: '#fef3c7',
    attentionColor: '#93c5fd',
    outputColor: '#fef3c7',
    concatNodeShape: 'sphere',
    attentionNodeShape: 'sphere',
    addFormulaText: 'Z^{ff}_{b,t,d}=H^{norm}_{cross,b,t,d}+H_{dec,b,t,d}',
    normFormulaText: 'H^{out}_{dec}=LayerNorm(Z^{ff})',
    outputFormulaText: 'B\\times T_{max}\\times d_{model}',
    lCols: 6,
    dRows: 6,
    bLayers: 3,
  },
};

const MODULE_FORMULAS = {
  x_rp: {
    input: '\\mathcal{F}_{rp}^{raw}',
    output: 'X_{rp} \\in \\mathbb{R}^{B\\times d_{rp}}',
  },
  x_spCtx: {
    input: '\\mathcal{C}_{sp}^{raw}',
    output: 'X_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{hist}}',
  },
  x_env: {
    input: '\\mathcal{F}_{env}^{raw}',
    output: 'X_{env} \\in \\mathbb{R}^{B\\times d_{env}}',
  },
  x_cand: {
    input: '\\mathcal{C}_{cand}^{raw}',
    output: 'X_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{cand}}',
  },
  inputEmbedding: {
    input: '\\{X_{rp}, X_{hist}, X_{env}, X_{cand}\\}',
    output: '\\{E_{rp}, E_{hist}, E_{env}, E_{cand}\\}',
  },
  concatPanel: {
    input: '\\{E_{rp}, E_{hist}, E_{env}, E_{cand}\\}',
    output: 'X_{enc} \\in \\mathbb{R}^{B\\times L\\times d}',
  },
  encoder: {
    input: 'X_{enc} \\in \\mathbb{R}^{B\\times L\\times d}',
    output: 'H_{enc} \\in \\mathbb{R}^{B\\times L\\times d}',
  },
  encoderAddNormBottom: {
    input: 'X_{enc},\\ H_{out} \\in \\mathbb{R}^{B\\times L\\times d}',
    output: 'H^{norm}=LayerNorm(X_{enc}+H_{out}),\\ Z_{b,l,d}=X_{b,l,d}+H_{out,b,l,d}',
  },
  encoderFeedForward: {
    input: 'H^{norm} \\in \\mathbb{R}^{B\\times L_{enc}\\times d_{model}}',
    output: 'FFN(H) \\in \\mathbb{R}^{B\\times L_{enc}\\times d_{model}}',
  },
  encoderAddNormTop: {
    input: 'H^{norm},\\ FFN(H) \\in \\mathbb{R}^{B\\times L_{enc}\\times d_{model}}',
    output: 'H_{enc}=LayerNorm(H^{norm}+FFN(H)) \\in \\mathbb{R}^{B\\times L_{enc}\\times d_{model}}',
  },
  outputEmbedding: {
    input: 'learned\\ query\\ slots',
    output: 'Q_{slot} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  questionSetAttention: {
    input: 'Q_{slot} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'H_{self} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  decoderAddNormSelf: {
    input: 'Q_{slot},\\ H_{self} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'H^{norm}_{self}=LayerNorm(Q_{slot}+H_{self}) \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  decoderCrossAttention: {
    input: 'H^{norm}_{self}, H_{enc}',
    output: 'O_{cross} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  decoderAddNormCross: {
    input: 'H^{norm}_{self},\\ O_{cross} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'O^{norm}_{cross}=LayerNorm(H^{norm}_{self}+O_{cross}) \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  decoder: {
    input: 'O^{norm}_{cross} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'H_{dec} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
  },
  decoderAddNormTop: {
    input: 'O^{norm}_{cross},\\ H_{dec} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'H^{out}_{dec} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}},\\ h_t = H^{out}_{dec}[:,t,:]',
  },
  maskHead: {
    input: 'h_t \\in \\mathbb{R}^{B\\times d_{model}}',
    output: 'm \\in \\{0,1\\}^{B\\times V}',
  },
  valueHead: {
    input: 'h_t \\in \\mathbb{R}^{B\\times d_{model}}',
    output: 'v \\in \\mathbb{R}^{B\\times V}',
  },
  scoreHead: {
    input: 'H_{slot} \\in \\mathbb{R}^{B\\times T_{max}\\times d_{model}}',
    output: 'score \\in \\mathbb{R}^{B\\times T_{max}\\times 1}',
  },
  countHead: {
    input: 'pool(H_{slot}) \\in \\mathbb{R}^{B\\times d_{model}}',
    output: 'p_{count} \\in \\mathbb{R}^{B\\times K_{count}}',
  },
  slotSelectHead: {
    input: 'H_{slot},\\ T_q',
    output: 'M_{slot} \\in \\{0,1\\}^{B\\times T_{max}}',
  },
  outputBlock: {
    input: '\\{T_q, M_{slot}, M_{var}, X_{eff}, score\\}',
    output: 'Q_{block} \\in \\mathbb{R}^{B\\times T_q\\times V}',
  },
  weights: {
    input: '\\nabla_{\\theta} L_{total}',
    output: '\\theta_{k+1}',
  },
  analysis: {
    input: '\\{a, y, notes\\}',
    output: '\\mathcal{M}_{batch}',
  },
};

function createOverlaySprite(text, {
  width = 320,
  height = 90,
  fontSize = 28,
  scale = 0.2,
  fill = '#e5eefc',
  background = 'rgba(8,14,24,0.68)',
  border = 'rgba(148,163,184,0.42)',
} = {}) {
  const sprite = createTextSprite(text, {
    width,
    height,
    fontSize,
    scale,
    fill,
    background,
    border,
  });
  sprite.material.depthTest = false;
  sprite.material.depthWrite = false;
  sprite.renderOrder = 34;
  return sprite;
}

function escapeHtml(text = '') {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeFormulaSyntax(text = '') {
  return String(text)
    .replace(/X_hist/g, 'X_{hist}')
    .replace(/X_sp_var/g, 'X_{spVar}')
    .replace(/E_sp_var/g, 'E_{spVar}')
    .replace(/X_stats/g, 'X_{env}')
    .replace(/\\mathbb\{R\}/g, 'ℝ')
    .replace(/\\mathcal\{F\}/g, 'ℱ')
    .replace(/\\mathcal\{C\}/g, '𝒞')
    .replace(/\\mathcal\{M\}/g, 'ℳ')
    .replace(/\\theta/g, 'θ')
    .replace(/\\lambda/g, 'λ')
    .replace(/\\partial/g, '∂')
    .replace(/\\nabla/g, '∇')
    .replace(/\\widetilde/g, '~')
    .replace(/\\times/g, '×')
    .replace(/\\to/g, '→')
    .replace(/\\in/g, '∈')
    .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
    .replace(/\\\{/g, '{')
    .replace(/\\\}/g, '}');
}

function tokenizeFormula(formula = '') {
  const text = normalizeFormulaSyntax(formula);
  const tokens = [];
  let buffer = '';

  const flush = () => {
    if (!buffer) return;
    tokens.push({ type: 'plain', text: buffer });
    buffer = '';
  };

  const readGroup = (source, startIndex) => {
    if (source[startIndex] === '{') {
      let depth = 1;
      let idx = startIndex + 1;
      let content = '';
      while (idx < source.length && depth > 0) {
        const ch = source[idx];
        if (ch === '{') {
          depth += 1;
          content += ch;
        } else if (ch === '}') {
          depth -= 1;
          if (depth > 0) content += ch;
        } else {
          content += ch;
        }
        idx += 1;
      }
      return { content, next: idx };
    }
    return {
      content: source[startIndex] || '',
      next: Math.min(source.length, startIndex + 1),
    };
  };

  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '_' || ch === '^') {
      flush();
      const type = ch === '_' ? 'sub' : 'sup';
      const { content, next } = readGroup(text, i + 1);
      tokens.push({ type, text: normalizeFormulaSyntax(content) });
      i = next;
      continue;
    }
    buffer += ch;
    i += 1;
  }
  flush();
  return tokens;
}

function formulaTextToHtml(text = '') {
  const isShapeSymbol = (part = '') => /^(B|V|h|1|\*|T_[A-Za-z]+|L_[A-Za-z]+|d_[A-Za-z]+|n_[A-Za-z]+|h_[A-Za-z]+|w_[A-Za-z0-9]+)$/.test(part);
  let normalized = normalizeFormulaSyntax(text);
  // Backward compatibility for legacy notation: R^[B, d_rp]
  normalized = normalized.replace(/R\^\[([^[\]]+)\]/g, (match, inner) => {
    const shape = inner
      .split(/\s*,\s*/)
      .filter(Boolean)
      .join('\\times');
    return `\\mathbb{R}^{${shape}}`;
  });
  normalized = normalized.replace(/\[([^[\]]+)\]/g, (match, inner) => {
    const parts = inner.split(/\s*,\s*/);
    if (!parts.length || !parts.every(isShapeSymbol)) return match;
    return `ℝ^{${parts.join('\\times')}}`;
  });
  normalized = normalized.replace(
    /([A-Za-zℱ𝒞ℳθλ∂∇]+(?:_\{[^}]+\}|_[A-Za-z0-9]+)?)\s+ℝ\^\{([^}]+)\}/g,
    '$1 \\in \\mathbb{R}^{$2}',
  );
  const tokensToHtml = (tokens) => tokens.map((token) => {
    if (token.type === 'plain') return escapeHtml(token.text);
    const inner = tokensToHtml(tokenizeFormula(token.text));
    return token.type === 'sub' ? `<sub>${inner}</sub>` : `<sup>${inner}</sup>`;
  }).join('');
  const lines = String(normalized).split('\n');
  const convertLine = (line) => {
    return tokensToHtml(tokenizeFormula(line));
  };
  return lines.map(convertLine).join('<br>');
}

function setMathBlock(target, text) {
  if (!target) return;
  target.classList.add('math-block');
  target.innerHTML = formulaTextToHtml(text);
}

function measureFormulaWidthFromTokens(ctx, tokens, fontSize = 22) {
  let width = 0;
  tokens.forEach((token) => {
    if (token.type === 'plain') {
      const size = fontSize;
      ctx.font = `600 ${size}px "Cambria Math", "STIX Two Math", "Times New Roman", serif`;
      width += ctx.measureText(token.text).width + 2;
      return;
    }
    const scriptSize = fontSize * 0.68;
    const nested = tokenizeFormula(token.text);
    width += measureFormulaWidthFromTokens(ctx, nested, scriptSize) + 1;
  });
  return width;
}

function measureFormulaWidth(ctx, formula, fontSize = 22) {
  return measureFormulaWidthFromTokens(ctx, tokenizeFormula(formula), fontSize);
}

function drawFormulaTokens(ctx, tokens, x, y, {
  fontSize = 22,
  color = '#f8fafc',
} = {}) {
  let cursorX = x;
  ctx.fillStyle = color;
  ctx.textBaseline = 'alphabetic';
  tokens.forEach((token) => {
    if (token.type === 'plain') {
      const size = fontSize;
      ctx.font = `600 ${size}px "Cambria Math", "STIX Two Math", "Times New Roman", serif`;
      ctx.fillText(token.text, cursorX, y);
      cursorX += ctx.measureText(token.text).width + 2;
      return;
    }
    const scriptSize = fontSize * 0.68;
    const offsetY = token.type === 'sub' ? fontSize * 0.34 : -fontSize * 0.42;
    const nested = tokenizeFormula(token.text);
    const nestedWidth = drawFormulaTokens(ctx, nested, cursorX, y + offsetY, {
      fontSize: scriptSize,
      color,
    });
    cursorX += nestedWidth + 1;
  });
  return cursorX - x;
}

function drawFormulaLine(ctx, formula, x, y, {
  fontSize = 22,
  color = '#f8fafc',
  align = 'left',
} = {}) {
  const tokens = tokenizeFormula(formula);
  const totalWidth = measureFormulaWidthFromTokens(ctx, tokens, fontSize);
  let cursorX = x;
  if (align === 'center') cursorX -= totalWidth / 2;
  if (align === 'right') cursorX -= totalWidth;
  drawFormulaTokens(ctx, tokens, cursorX, y, { fontSize, color });
}

function createFormulaSprite(formula, {
  width = 520,
  height = 92,
  fontSize = 20,
  scale = 0.18,
  align = 'left',
  fill = '#f8fafc',
  background = 'rgba(8,14,24,0.56)',
  border = 'rgba(148,163,184,0.18)',
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  material.depthTest = false;
  material.depthWrite = false;
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width * scale, height * scale, 1);
  sprite.renderOrder = 35;

  function draw(nextFormula, overrides = {}) {
    const nextAlign = overrides.align || align;
    const nextFill = overrides.fill || fill;
    const nextBackground = overrides.background || background;
    const nextBorder = overrides.border || border;
    const nextFontSize = overrides.fontSize || fontSize;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = nextBackground;
    ctx.strokeStyle = nextBorder;
    ctx.lineWidth = 3;
    drawRoundRect(ctx, 3, 3, width - 6, height - 6, 18);
    ctx.fill();
    ctx.stroke();
    const anchorX = nextAlign === 'right' ? width - 20 : (nextAlign === 'center' ? width / 2 : 20);
    drawFormulaLine(ctx, nextFormula, anchorX, height / 2 + nextFontSize * 0.18, {
      fontSize: nextFontSize,
      color: nextFill,
      align: nextAlign,
    });
    texture.needsUpdate = true;
  }

  draw(formula);
  sprite.userData.updateFormula = draw;
  return sprite;
}

// 当前 Three.js 场景：x 是左右，y 是上下，z 是前后深度。
// X_hist / X_cand 以及 Input Embedding 里的 ctx/cand 使用 xz 平面绘制 Lxd：d -> x，L -> -z，y 固定。
const HISTORICAL_INPUT_MATRIX_Z = 8;
const HISTORICAL_BATCH_SHIFT_X = -9;
const HISTORICAL_BATCH_SHIFT_Z = -46;

function addVariableModuleVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const cubeSize = Math.min(16, spec.size.y * 0.22);
  const startX = -spec.size.x * 0.24;
  const stepX = spec.size.x * 0.18;
  const cubeY = -spec.size.y * 0.02;
  const labelY = cubeY - cubeSize * 1.45;
  const cubeColor = lerpColorHex(visualSpec.tint || spec.haloColor || '#7dd3fc', '#ffffff', 0.18);
  const isHistoricalMatrix = moduleEntry.id === 'x_spCtx' || moduleEntry.id === 'x_cand';

  const cubes = [];
  if (isHistoricalMatrix) {
    const lCount = 3;
    const dCount = moduleEntry.id === 'x_cand' ? 5 : 4;
    const matrixWidth = spec.size.x * 0.58;
    const matrixDepth = spec.size.y * 0.46;
    const cellSize = Math.min(12, cubeSize * 0.78);
    const fixedY = spec.size.y * 0.08;
    const matrixOriginZ = HISTORICAL_INPUT_MATRIX_Z;

    for (let lIdx = 0; lIdx < lCount; lIdx += 1) {
      for (let dIdx = 0; dIdx < dCount; dIdx += 1) {
        const dRatio = dCount === 1 ? 0.5 : dIdx / (dCount - 1);
        const lRatio = lCount === 1 ? 0 : lIdx / (lCount - 1);
        // d 沿 x 展开；L 沿 -z 展开；y 固定，形成 xz 平面内的 Lxd 矩阵。
        const x = (dRatio - 0.5) * matrixWidth;
        const z = matrixOriginZ - lRatio * matrixDepth;

        const cube = new THREE.Mesh(
          new THREE.BoxGeometry(cellSize, cellSize, cellSize),
          new THREE.MeshPhongMaterial({
            color: cubeColor,
            transparent: true,
            opacity: 0.48,
            emissive: new THREE.Color(cubeColor),
            emissiveIntensity: 0.12,
            shininess: 74,
          }),
        );
        cube.position.set(x, fixedY, z);
        interior.add(cube);
        cubes.push(cube);

        // Batch B：y 不变，只沿 -x 和 -z 方向偏移。
        [1, 2, 3].forEach((batchIdx) => {
          const ghost = new THREE.Mesh(
            new THREE.BoxGeometry(cellSize * 0.94, cellSize * 0.94, cellSize * 0.94),
            new THREE.MeshPhongMaterial({
              color: cubeColor,
              transparent: true,
              opacity: batchIdx === 1 ? 0.2 : (batchIdx === 2 ? 0.12 : 0.07),
              emissive: new THREE.Color(cubeColor),
              emissiveIntensity: 0.05,
              shininess: 40,
              depthWrite: false,
            }),
          );
          ghost.position.set(
            x + HISTORICAL_BATCH_SHIFT_X * batchIdx,
            fixedY,
            z + HISTORICAL_BATCH_SHIFT_Z * batchIdx,
          );
          interior.add(ghost);
        });
      }
    }

    const dLabel = createFormulaSprite('d', {
      width: 72,
      height: 40,
      fontSize: 18,
      scale: 0.1,
      align: 'center',
      fill: '#bfdbfe',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    dLabel.position.set(matrixWidth * 0.64, fixedY, matrixOriginZ);
    interior.add(dLabel);

    const seqSymbol = moduleEntry.id === 'x_spCtx' ? 'L_{hist}' : 'L_{cand}';
    const lLabel = createFormulaSprite(seqSymbol, {
      width: 130,
      height: 40,
      fontSize: 16,
      scale: 0.1,
      align: 'center',
      fill: '#fde68a',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    lLabel.position.set(-matrixWidth * 0.66, fixedY, matrixOriginZ - matrixDepth * 0.5);
    interior.add(lLabel);

    group.add(interior);
    moduleAnimators.push((elapsed) => {
      cubes.forEach((cube, index) => {
        const s = 1 + Math.sin(elapsed * 1.85 + index * 0.42 + spec.position.x * 0.0018) * 0.045;
        cube.scale.setScalar(s);
        // 只做缩放呼吸，不改变 z，避免矩阵上下位置漂移。
      });
    });
    moduleEntry.interior = interior;
    return;
  }

  visualSpec.variables.forEach((name, index) => {
    const x = startX + index * stepX;
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize),
      new THREE.MeshPhongMaterial({
        color: cubeColor,
        transparent: true,
        opacity: 0.42,
        emissive: new THREE.Color(cubeColor),
        emissiveIntensity: 0.12,
        shininess: 74,
      }),
    );
    cube.position.set(x, cubeY, 10);
    interior.add(cube);
    cubes.push(cube);

    const label = createOverlaySprite(name, {
      width: 160,
      height: 52,
      fontSize: 18,
      scale: 0.12,
      fill: '#dbeafe',
      background: 'rgba(15,23,42,0.34)',
      border: 'rgba(148,163,184,0.16)',
    });
    label.position.set(x, labelY, 12);
    interior.add(label);

    if (index === visualSpec.variables.length - 1 && visualSpec.variables.length > 1) {
      const ellipsis = createOverlaySprite('...', {
        width: 84,
        height: 44,
        fontSize: 28,
        scale: 0.11,
        fill: '#f8fafc',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      ellipsis.position.set(x - stepX * 0.5, cubeY + 1, 12);
      interior.add(ellipsis);
    }
  });

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    cubes.forEach((cube, index) => {
      const s = 1 + Math.sin(elapsed * 1.8 + index * 0.75 + spec.position.x * 0.002) * 0.05;
      cube.scale.setScalar(s);
      cube.position.z = 10 + Math.sin(elapsed * 1.4 + index * 0.55) * 1.8;
    });
  });
  moduleEntry.interior = interior;
}

function addEmbeddingNetworkVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  // Input Embedding 使用单层线性投影：输入层 -> 输出层（无隐藏层）。
  const layerYs = [-spec.size.y * 0.16, spec.size.y * 0.2];
  // z 是上下轴；ctx/cand 的输入矩阵必须和下方 X_hist/X_cand 模块保持同一个 z 平面。
  const layerZs = [HISTORICAL_INPUT_MATRIX_Z, 22];
  const networkSpecs = visualSpec.networks || [];
  const networkCount = Math.max(1, networkSpecs.length || 4);
  const spanWidth = spec.size.x * 0.86;
  const networkSpacing = spanWidth / networkCount;
  const networkHalfWidth = Math.min(30, networkSpacing * 0.3);
  const nodeRadius = 5.6;
  const layers = [];

  const getCondensedLayerLayout = (count, halfWidth) => {
    if (count <= 4) {
      const xs = Array.from({ length: count }, (_, i) => {
        const t = count === 1 ? 0.5 : i / (count - 1);
        return (t - 0.5) * halfWidth * 2;
      });
      const displayIndices = Array.from({ length: count }, (_, i) => i + 1);
      return { xs, displayIndices, ellipsisX: null };
    }
    const left = -halfWidth * 0.92;
    const mid1 = -halfWidth * 0.45;
    const mid2 = -halfWidth * 0.08;
    const right = halfWidth * 0.92;
    // Condensed display: first few nodes + last node.
    const displayIndices = [1, 2, 3, count];
    return {
      xs: [left, mid1, mid2, right],
      displayIndices,
      ellipsisX: (mid2 + right) * 0.5,
    };
  };

  const resolveInputLayerNodeCount = (net, fallbackCount = 4) => {
    // Prefer explicit numeric config when provided.
    const explicitCount = Number(net.inputDisplayCount);
    if (Number.isFinite(explicitCount) && explicitCount > 0) {
      return Math.max(1, Math.round(explicitCount));
    }
    // Otherwise align the first-layer node count with source input dimensionality.
    const sourceVars = MODULE_INTERIOR_SPECS[net.sourceId]?.variables;
    if (Array.isArray(sourceVars) && sourceVars.length > 0) {
      return sourceVars.length;
    }
    const inputDimAsNumber = Number(net.inputDim);
    if (Number.isFinite(inputDimAsNumber) && inputDimAsNumber > 0) {
      return Math.max(1, Math.round(inputDimAsNumber));
    }
    const inputCountAsNumber = Number(net.inputCount);
    if (Number.isFinite(inputCountAsNumber) && inputCountAsNumber > 0) {
      return Math.max(1, Math.round(inputCountAsNumber));
    }
    return Math.max(1, Math.round(Number(fallbackCount) || 4));
  };

  const resolvedNetworks = networkSpecs.length ? networkSpecs : [
    { key: 'rp', sourceId: 'x_rp', title: 'E_rp', inputCount: 'd_{rp}', hiddenCount: 'h_{rp}', outputCount: 'd_{model}', sequenceLength: '1', inputDim: 'd_{rp}', hiddenDim: 'h_{rp}', outputDim: 'd_{model}', color: '#61dafb', displayCounts: [9, 7] },
    { key: 'spCtx', sourceId: 'x_spCtx', title: 'E_hist', inputCount: 'd_{hist}', hiddenCount: 'h_{spCtx}', outputCount: 'd_{model}', sequenceLength: 'L_{hist}', inputDim: 'd_{hist}', hiddenDim: 'h_{ctx}', outputDim: 'd_{model}', color: '#fbbf24', displayCounts: [7, 8] },
    { key: 'env', sourceId: 'x_env', title: 'E_env', inputCount: 'd_{env}', hiddenCount: 'h_{env}', outputCount: 'd_{model}', sequenceLength: '1', inputDim: 'd_{env}', hiddenDim: 'h_{env}', outputDim: 'd_{model}', color: '#7ee787', displayCounts: [6, 7] },
    { key: 'cand', sourceId: 'x_cand', title: 'E_cand', inputCount: 'd_{cand}', hiddenCount: 'h_{cand}', outputCount: 'd_{model}', sequenceLength: 'L_{cand}', inputDim: 'd_{cand}', hiddenDim: 'h_{cand}', outputDim: 'd_{model}', color: '#a78bfa', displayCounts: [10, 8] },
  ];

  resolvedNetworks.forEach((net, netIdx) => {
    const centerX = -spanWidth * 0.5 + networkSpacing * (netIdx + 0.5);
    const perNetLayers = [];
    const baseDisplayCounts = Array.isArray(net.displayCounts) ? net.displayCounts : [8, 6];
    const displayCounts = [
      resolveInputLayerNodeCount(net, baseDisplayCounts[0]),
      Math.max(1, Number(baseDisplayCounts[1] || 6)),
    ];

    const netTitle = createOverlaySprite(net.sourceId || `x_${net.key}`, {
      width: 180,
      height: 50,
      fontSize: 20,
      scale: 0.11,
      fill: '#dbeafe',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    netTitle.position.set(centerX, layerYs[0] - 24, 12);
    interior.add(netTitle);
    const seqLenSymbol = net.sequenceLength || '1';
    const branchShapeFormula = `${net.title || `E_${net.key}`} \\in \\mathbb{R}^{B\\times ${seqLenSymbol}\\times d_{model}}`;
    const branchShape = createFormulaSprite(branchShapeFormula, {
      width: 380,
      height: 54,
      fontSize: 14,
      scale: 0.085,
      align: 'center',
      fill: '#cbd5e1',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    branchShape.position.set(centerX, layerYs[layerYs.length - 1] + 22, 12);
    interior.add(branchShape);

    const showBatchLabel = !(net.key === 'spCtx' || net.key === 'cand');
    if (showBatchLabel) {
      const batchLabel = createFormulaSprite('B\\times', {
        width: 130,
        height: 44,
        fontSize: 17,
        scale: 0.085,
        align: 'center',
        fill: '#bfdbfe',
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      batchLabel.position.set(centerX - networkHalfWidth - 24, layerYs[0] - 6, 12);
      interior.add(batchLabel);
    }

    layerYs.forEach((layerY, layerIdx) => {
      const fullCount = Math.max(1, Number(displayCounts[layerIdx] || 4));
      const totalLayers = layerYs.length;
      const layerZ = layerZs[Math.min(layerIdx, layerZs.length - 1)];
      const countSymbols = [
        net.inputCount || net.inputDim || `d_{${net.key}}`,
        net.outputCount || net.outputDim || 'd_{model}',
      ];
      const isSequenceMatrixLayer = net.key === 'spCtx' || net.key === 'cand';
      let nodeSlots = [];
      let displayIndices = [];
      let ellipsisX = null;
      let matrixGridDepth = 0;
      if (isSequenceMatrixLayer) {
        const rows = 3;
        const cols = layerIdx === 0
          ? (net.key === 'cand' ? 5 : 4)
          : 5;
        const gridWidth = networkHalfWidth * 1.9;
        matrixGridDepth = 34;
        for (let r = 0; r < rows; r += 1) {
          for (let c = 0; c < cols; c += 1) {
            const tx = cols === 1 ? 0.5 : c / (cols - 1);
            const ty = rows === 1 ? 0.5 : r / (rows - 1);
            nodeSlots.push({
              x: (tx - 0.5) * gridWidth,
              // Input Embedding 的 ctx/cand 输入和输出都放在 xz 平面：d/d_model -> x，L -> -z，y 固定。
              y: 0,
              z: -ty * matrixGridDepth,
              row: r,
              col: c,
              rows,
              cols,
            });
            displayIndices.push(`${r + 1},${c + 1}`);
          }
        }
        ellipsisX = gridWidth * 0.64;
        const matrixLabel = createFormulaSprite(`${seqLenSymbol}`, {
          width: 180,
          height: 42,
          fontSize: 14,
          scale: 0.08,
          align: 'center',
          fill: '#bfdbfe',
          background: 'rgba(0,0,0,0)',
          border: 'rgba(0,0,0,0)',
        });
        matrixLabel.position.set(centerX, layerY, layerZ - matrixGridDepth * 0.95);
        interior.add(matrixLabel);
      } else {
        const condensed = getCondensedLayerLayout(fullCount, networkHalfWidth);
        nodeSlots = condensed.xs.map((x) => ({ x, y: 0 }));
        displayIndices = condensed.displayIndices;
        ellipsisX = condensed.ellipsisX;
      }
      const nodes = [];
      nodeSlots.forEach((slot, nodeIdx) => {
        const neuronDisplayIndex = displayIndices?.[nodeIdx] || (nodeIdx + 1);
        const nodeGeometry = isSequenceMatrixLayer
          ? new THREE.SphereGeometry(nodeRadius * 0.58, 20, 20)
          : new THREE.SphereGeometry(nodeRadius, 20, 20);
        const node = new THREE.Mesh(
          nodeGeometry,
          new THREE.MeshToonMaterial({
            color: net.color || '#61dafb',
            transparent: true,
            opacity: 0.95,
          }),
        );
        node.position.set(
          centerX + slot.x,
          layerY + (slot.y || 0),
          layerZ + (slot.z || 0),
        );
        interior.add(node);
        nodes.push(node);

        const prevCountSymbol = layerIdx > 0 ? countSymbols[layerIdx - 1] : null;
        const currCountSymbol = countSymbols[layerIdx];
        const nextCountSymbol = layerIdx < totalLayers - 1 ? countSymbols[layerIdx + 1] : null;
        const weights = Array.from({ length: Math.min(6, Math.max(3, nodeSlots.length + 1)) }, (_, wIdx) => {
          const raw = Math.sin((netIdx + 1) * 19 + (layerIdx + 1) * 11 + (nodeIdx + 1) * 7 + (wIdx + 1) * 5) * 0.41;
          return Number(raw.toFixed(3));
        });
        const currentNodeFormula = isSequenceMatrixLayer
          ? (layerIdx === 0
            ? `x_{${net.key}} \\in \\mathbb{R}^{B\\times ${seqLenSymbol}\\times ${currCountSymbol}}`
            : `${net.title || `E_{${net.key}}`} \\in \\mathbb{R}^{B\\times ${seqLenSymbol}\\times ${currCountSymbol}}`)
          : (layerIdx === 0
            ? `x_{${net.key}} \\in \\mathbb{R}^{${currCountSymbol}}`
            : `e_{${net.key}} \\in \\mathbb{R}^{${currCountSymbol}}`);
        const upperWeightFormula = layerIdx > 0
          ? `W_{${net.key}}^{(${layerIdx})} \\in \\mathbb{R}^{${prevCountSymbol}\\times${currCountSymbol}}`
          : null;
        const lowerWeightFormula = layerIdx < totalLayers - 1
          ? `W_{${net.key}}^{(${layerIdx + 1})} \\in \\mathbb{R}^{${currCountSymbol}\\times${nextCountSymbol}}`
          : null;
        node.userData.embeddingNode = {
          networkKey: net.key,
          networkSource: net.sourceId,
          networkTitle: net.title,
          layerIndex: layerIdx,
          neuronIndex: nodeIdx,
          neuronDisplayIndex,
          layerDimensionDisplay: isSequenceMatrixLayer ? `B\\times ${seqLenSymbol}\\times ${currCountSymbol}` : countSymbols[layerIdx],
          matrixCoord: isSequenceMatrixLayer ? {
            row: slot.row + 1,
            col: slot.col + 1,
            rows: slot.rows,
            cols: slot.cols,
            isLastRow: slot.row === slot.rows - 1,
            isLastCol: slot.col === slot.cols - 1,
            sequenceLength: seqLenSymbol,
            inputDim: currCountSymbol,
          } : null,
          isLastVisibleNode: !isSequenceMatrixLayer && nodeIdx === (nodeSlots.length - 1),
          inputFormula: layerIdx === 0
            ? `x_{${net.key}} \\in \\mathbb{R}^{${countSymbols[0]}}`
            : `x_{${net.key}} \\in \\mathbb{R}^{${prevCountSymbol}}`,
          outputFormula: layerIdx === totalLayers - 1
            ? branchShapeFormula
            : `e_{${net.key}} \\in \\mathbb{R}^{${nextCountSymbol}}`,
          branchOutputShape: branchShapeFormula,
          weightFormula: lowerWeightFormula || upperWeightFormula || `W_{${net.key}} \\in \\mathbb{R}^{*}`,
          currentNodeFormula,
          upperWeightCount: layerIdx > 0 ? prevCountSymbol : null,
          upperWeightFormula,
          lowerWeightCount: layerIdx < totalLayers - 1 ? nextCountSymbol : null,
          lowerWeightFormula,
          weights,
          layerFullCount: fullCount,
          layerVisibleCount: nodeSlots.length,
        };
        embeddingNeuronTargets.push(node);
      });

      if (ellipsisX !== null) {
        const ellipsis = createOverlaySprite('...', {
          width: 56,
          height: 36,
          fontSize: 24,
          scale: 0.1,
          fill: '#f8fafc',
          background: 'rgba(0,0,0,0)',
          border: 'rgba(0,0,0,0)',
        });
        ellipsis.position.set(centerX + ellipsisX, layerY, layerZ);
        interior.add(ellipsis);
        if (isSequenceMatrixLayer) {
          const ellipsis2 = createOverlaySprite('...', {
            width: 56,
            height: 36,
            fontSize: 24,
            scale: 0.1,
            fill: '#f8fafc',
            background: 'rgba(0,0,0,0)',
            border: 'rgba(0,0,0,0)',
          });
          ellipsis2.position.set(centerX, layerY, layerZ - matrixGridDepth * 0.68);
          interior.add(ellipsis2);
        }
      }

      perNetLayers.push(nodes);
      layers.push(nodes);
    });

    for (let layerIdx = 0; layerIdx < perNetLayers.length - 1; layerIdx += 1) {
      perNetLayers[layerIdx].forEach((fromNode) => {
        perNetLayers[layerIdx + 1].forEach((toNode) => {
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
              fromNode.position.clone(),
              toNode.position.clone(),
            ]),
            new THREE.LineBasicMaterial({
              color: net.color || '#93c5fd',
              transparent: true,
              opacity: 0.32,
            }),
          );
          interior.add(line);
        });
      });
    }

    // Batch B 可视化：
    // - spCtx/cand：输入层按 xz 平面的 Lxd 矩阵显示，batch 沿 -x / -z 偏移；
    // - y 是当前场景的上下方向，因此 matrix 和 batch 都不改变 y。
    if (net.key === 'spCtx' || net.key === 'cand') {
      const replicaOffsets = [
        { dx: HISTORICAL_BATCH_SHIFT_X, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z, opacity: 0.22 },
        { dx: HISTORICAL_BATCH_SHIFT_X * 2, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z * 2, opacity: 0.15 },
        { dx: HISTORICAL_BATCH_SHIFT_X * 3, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z * 3, opacity: 0.09 },
      ];
      replicaOffsets.forEach((replica) => {
        perNetLayers.forEach((layerNodes) => {
          layerNodes.forEach((node) => {
            const ghostNode = new THREE.Mesh(
              new THREE.SphereGeometry(nodeRadius * 0.58 * 0.9, 14, 14),
              new THREE.MeshToonMaterial({
                color: net.color || '#93c5fd',
                transparent: true,
                opacity: replica.opacity,
                depthWrite: false,
              }),
            );
            ghostNode.position.copy(node.position).add(new THREE.Vector3(replica.dx, 0, replica.dz));
            interior.add(ghostNode);
          });
        });
      });
    } else {
      // x_rp / x_env 是单 token 输入；batch 只沿 -x 和 z 深度方向延伸，不再改变 y 上下位置。
      const replicaOffsets = [
        { dx: -12, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z, opacity: 0.22 },
        { dx: -24, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z * 2, opacity: 0.15 },
        { dx: -36, dy: 0, dz: HISTORICAL_BATCH_SHIFT_Z * 3, opacity: 0.09 },
      ];
      replicaOffsets.forEach((replica) => {
        perNetLayers.forEach((layerNodes) => {
          layerNodes.forEach((node) => {
            const ghostNode = new THREE.Mesh(
              new THREE.SphereGeometry(nodeRadius * 0.84, 14, 14),
              new THREE.MeshToonMaterial({
                color: net.color || '#93c5fd',
                transparent: true,
                opacity: replica.opacity,
                depthWrite: false,
              }),
            );
            ghostNode.position.copy(node.position).add(new THREE.Vector3(replica.dx, 0, replica.dz));
            interior.add(ghostNode);
          });
        });
      });
    }

  });

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    layers.flat().forEach((node, index) => {
      const s = 1 + Math.sin(elapsed * 2.2 + index * 0.4) * 0.08;
      node.scale.setScalar(s);
    });
  });
  moduleEntry.interior = interior;
  moduleEntry.embeddingLayers = layers;
}

function buildConcatCubeDimensionInfo(info, { stage = 'Concat Block 悬停' } = {}) {
  const tokenFormula = info?.tokenFormula || info?.key || 'E_{*}';
  const shapeFormula = info?.shapeFormula || 'E_{*} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}';
  const sourceFormula = info?.sourceFormula || '';
  const sourceBlockFormula = info?.sourceBlockFormula || '';
  const sourceShapeFormula = info?.sourceShapeFormula || '';
  const tokenCountFormula = info?.tokenCountFormula || '1';
  const cellText = info?.sequenceIndex
    ? `\n可视化单元：sequence ${info.sequenceIndex} / ${tokenCountFormula}, d_model row ${info.dModelRow || 1}`
    : '';
  const tail = info?.isOutputMatrix
    ? '该方块是 concat 后 X_enc 输出矩阵中的一个单元；颜色表示它来自哪个 embedding 段。'
    : info?.isFinal
      ? '最后一个立方体表示 concat 后编码序列的总维度。'
    : '该方块属于 concat 输出矩阵中的一个 embedding block；同色块来自同一个 Input Embedding 输出。';
  return {
    module: `Concat, Block ${tokenFormula}`,
    stage,
    before: `${sourceFormula ? `来源:\n${sourceFormula}\n` : ''}${sourceBlockFormula ? `来源段:\n${sourceBlockFormula}\n` : ''}${sourceShapeFormula ? `来源段维度:\n${sourceShapeFormula}\n` : ''}Block:\n${tokenFormula}`,
    after: `维度:\n${shapeFormula}`,
    detail: `${tail}${cellText}\n主矩阵是 L_{sum} × d，x 方向表示 L，z 方向表示 d；B 是第一维，用整张矩阵沿 y 方向向下的半透明层叠表达。`,
  };
}

function applyConcatCubeDimensionInfo(info, { stage = 'Concat Block 悬停' } = {}) {
  dashboardState.focusId = 'concatPanel';
  dashboardState.dimensionInfo = buildConcatCubeDimensionInfo(info, { stage });
  updateDashboard();
}

function setConcatCubeTooltipContent(info) {
  if (!info) {
    embeddingTooltip.style.display = 'none';
    return;
  }
  const tokenHtml = formulaTextToHtml(info.tokenFormula || info.key || 'E_{*}');
  const dimHtml = formulaTextToHtml(info.shapeFormula || '');
  const sourceHtml = formulaTextToHtml(info.sourceFormula || '');
  const sourceBlockHtml = formulaTextToHtml(info.sourceBlockFormula || '');
  const sourceShapeHtml = formulaTextToHtml(info.sourceShapeFormula || '');
  const headText = info.isFinal ? 'Concat 结果维度' : 'Concat 矩阵块';
  embeddingTooltip.innerHTML = [
    `<div style="color:#fbbf24;font-weight:700;margin-bottom:3px;">${headText}</div>`,
    sourceHtml ? `<div>来源: <span style="color:#a7f3d0">${sourceHtml}</span></div>` : '',
    sourceBlockHtml ? `<div>来源段: <span style="color:#a7f3d0">${sourceBlockHtml}</span></div>` : '',
    sourceShapeHtml ? `<div>来源段维度: <span style="color:#cbd5e1">${sourceShapeHtml}</span></div>` : '',
    `<div>Block: <span style="color:#a7f3d0">${tokenHtml}</span></div>`,
    `<div>维度: <span style="color:#93c5fd">${dimHtml}</span></div>`,
    info.sequenceIndex ? `<div>单元: <span style="color:#cbd5e1">L ${info.sequenceIndex} / ${info.tokenCountFormula}, d row ${info.dModelRow}</span></div>` : '',
    '<div style="margin-top:3px;color:#cbd5e1;">主矩阵=L_sum×d；x 表示 L，z 表示 d，B 维用整张矩阵沿 y 方向向下层叠表示</div>',
  ].join('');
  embeddingTooltip.style.display = 'block';
}

function clearHoveredConcatCube() {
  if (hoveredConcatCube?.mesh) {
    hoveredConcatCube.mesh.material.emissiveIntensity = hoveredConcatCube.baseEmissive;
    hoveredConcatCube.mesh.scale.copy(hoveredConcatCube.baseScale);
  }
  hoveredConcatCube = null;
}

function buildAttentionTensorDimensionInfo(info, { stage = 'Attention 球体维度' } = {}) {
  const bText = info?.isLastTensorCell ? info.bSymbol : `b=${info.bIndex}/${info.bLayers}`;
  const lText = info?.isLastTensorCell ? info.lSymbol : `l=${info.lIndex}/${info.lCols}`;
  const dText = info?.isLastTensorCell ? info.dSymbol : `d=${info.dIndex}/${info.dRows}`;
  const cellFormula = info?.isLastTensorCell
    ? `${info.tensorLabel} \\in \\mathbb{R}^{${info.bSymbol}\\times ${info.lSymbol}\\times ${info.dSymbol}}`
    : `${info.tensorLabel}_{${info.bIndex},${info.lIndex},${info.dIndex}}`;
  const detail = info?.isLastTensorCell
    ? `这是该张量可视化里的最后一个球体，用它标记整体维度：第一维是 batch ${info.bSymbol}，第二维是 ${info.lSymbol}，第三维是 ${info.dSymbol}。`
    : `该球体表示 ${info.bSymbol}×${info.lSymbol}×${info.dSymbol} 三维张量中的一个标量单元；同一层内的球体组成 ${info.lSymbol}×${info.dSymbol} 矩阵，沿 y 方向向下堆叠表示 batch ${info.bSymbol}。`;
  return {
    module: `${info.tensorLabel}, ${bText}, ${lText}, ${dText}`,
    stage,
    before: `张量：\n${info.tensorShapeFormula}`,
    after: `当前球体：\n${cellFormula}`,
    detail,
  };
}

function applyAttentionTensorDimensionInfo(info, { stage = 'Attention 球体维度' } = {}) {
  dashboardState.focusId = info?.moduleId || 'encoder';
  dashboardState.dimensionInfo = buildAttentionTensorDimensionInfo(info, { stage });
  updateDashboard();
}

function setAttentionTensorTooltipContent(info) {
  if (!info) {
    embeddingTooltip.style.display = 'none';
    return;
  }
  const shapeHtml = formulaTextToHtml(info.tensorShapeFormula || '');
  const currentHtml = formulaTextToHtml(
    info.isLastTensorCell
      ? `${info.tensorLabel}\\in\\mathbb{R}^{${info.bSymbol}\\times ${info.lSymbol}\\times ${info.dSymbol}}`
      : `${info.tensorLabel}_{${info.bIndex},${info.lIndex},${info.dIndex}}`,
  );
  embeddingTooltip.innerHTML = [
    `<div style="color:#fbbf24;font-weight:700;margin-bottom:3px;">${info.tensorStage || 'Attention Tensor'}</div>`,
    `<div>张量: <span style="color:#93c5fd">${shapeHtml}</span></div>`,
    `<div>当前球体: <span style="color:#a7f3d0">${currentHtml}</span></div>`,
    `<div>位置: <span style="color:#cbd5e1">B ${info.bIndex}/${info.bLayers}, L ${info.lIndex}/${info.lCols}, d ${info.dIndex}/${info.dRows}</span></div>`,
    info.isLastTensorCell
      ? '<div style="margin-top:3px;color:#fde68a;">最后一个球体表示整体维度 B×L×d。</div>'
      : '<div style="margin-top:3px;color:#cbd5e1;">单个球体是张量中的一个标量单元。</div>',
  ].join('');
  embeddingTooltip.style.display = 'block';
}

function clearHoveredAttentionTensorNode() {
  if (hoveredAttentionTensorNode?.mesh) {
    hoveredAttentionTensorNode.mesh.material.emissiveIntensity = hoveredAttentionTensorNode.baseEmissive;
    hoveredAttentionTensorNode.mesh.scale.copy(hoveredAttentionTensorNode.baseScale);
  }
  hoveredAttentionTensorNode = null;
}

function handleAttentionTensorHover(nodeMesh, { selected = false } = {}) {
  if (!nodeMesh?.userData?.attentionTensorNode) return;
  if (!selected && hoveredAttentionTensorNode?.mesh === nodeMesh) return;
  clearHoveredAttentionTensorNode();
  hoveredAttentionTensorNode = {
    mesh: nodeMesh,
    baseEmissive: nodeMesh.material.emissiveIntensity || 0,
    baseScale: nodeMesh.scale.clone(),
  };
  nodeMesh.material.emissiveIntensity = (nodeMesh.material.emissiveIntensity || 0) + (selected ? 0.42 : 0.26);
  nodeMesh.scale.setScalar(selected ? 1.22 : 1.12);
  const info = nodeMesh.userData.attentionTensorNode;
  setStatus('Attention Tensor', `查看 ${info.tensorLabel} 的球体维度：B ${info.bIndex}, L ${info.lIndex}, d ${info.dIndex}。`);
  setAttentionTensorTooltipContent(info);
  applyAttentionTensorDimensionInfo(info, { stage: selected ? 'Attention 球体点击' : 'Attention 球体悬停' });
}

function handleConcatCubeHover(cubeMesh) {
  if (!cubeMesh?.userData?.concatCube) return;
  if (hoveredConcatCube?.mesh === cubeMesh) return;
  clearHoveredConcatCube();
  hoveredConcatCube = {
    mesh: cubeMesh,
    baseEmissive: cubeMesh.material.emissiveIntensity || 0,
    baseScale: cubeMesh.scale.clone(),
  };
  cubeMesh.material.emissiveIntensity = (cubeMesh.material.emissiveIntensity || 0) + 0.24;
  cubeMesh.scale.set(1.1, 1.1, 1.1);
  const info = cubeMesh.userData.concatCube;
  setStatus('Concat Block', `正在查看 ${info.key} 的矩阵块维度与 concat 位置。`);
  setConcatCubeTooltipContent(info);
  applyConcatCubeDimensionInfo(info, { stage: 'Concat Block 悬停' });
}

function addConcatModuleVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const blocks = Array.isArray(visualSpec?.concatBlocks) && visualSpec.concatBlocks.length
    ? visualSpec.concatBlocks.filter((block) => !block.isFinal)
    : [];
  const finalBlock = Array.isArray(visualSpec?.concatBlocks)
    ? visualSpec.concatBlocks.find((block) => block.isFinal)
    : null;
  if (!blocks.length) {
    moduleEntry.interior = interior;
    group.add(interior);
    return;
  }

  const cubeSize = Math.min(7.2, spec.size.y * 0.18);
  const fixedY = -spec.size.y * 0.04;
  const matrixFrontZ = 18;
  const lStepX = cubeSize * 1.38;
  const dStepZ = cubeSize * 1.38;
  const blockGapX = 0;
  const dRows = 6;
  const batchStepY = cubeSize * 1.28;
  const totalTokenSlots = blocks.reduce((sum, block) => sum + Math.max(1, Number(block.displayTokens || 1)), 0);
  const totalWidth = totalTokenSlots * lStepX + Math.max(0, totalTokenSlots - 1) * blockGapX;
  const startX = -totalWidth / 2 + lStepX * 0.5;
  const fallbackColor = visualSpec.tint || spec.haloColor || '#93c5fd';
  const cubes = [];
  const remapCells = [];
  const outputFormula = finalBlock?.shapeFormula
    || 'X_{enc} \\in \\mathbb{R}^{B\\times (1+L_{hist}+1+L_{cand})\\times 6}';
  const outputTokenFormula = finalBlock?.tokenFormula || finalBlock?.key || 'X_{enc}';
  let globalTokenCursor = 0;

  blocks.forEach((block) => {
    const displayTokens = Math.max(1, Number(block.displayTokens || 1));
    const blockColor = lerpColorHex(block.color || fallbackColor, '#ffffff', 0.14);
    const blockStartX = startX + globalTokenCursor * lStepX;

    for (let lLocal = 0; lLocal < displayTokens; lLocal += 1) {
      const lIndex = globalTokenCursor + lLocal;
      for (let dIndex = 0; dIndex < dRows; dIndex += 1) {
        const x = startX + lIndex * lStepX;
        const z = matrixFrontZ - dIndex * dStepZ;
        const cube = new THREE.Mesh(
          new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize),
          new THREE.MeshPhongMaterial({
            color: blockColor,
            transparent: true,
            opacity: 0.54,
            emissive: new THREE.Color(blockColor),
            emissiveIntensity: 0.1,
            shininess: 70,
          }),
        );
        cube.position.set(x, fixedY, z);
        cube.userData.concatCube = {
          key: outputTokenFormula,
          tokenFormula: outputTokenFormula,
          sourceFormula: finalBlock?.sourceFormula || 'concat(E_{rp},E_{hist},E_{env},E_{cand})',
          sourceBlockFormula: block.tokenFormula || block.key,
          sourceShapeFormula: block.shapeFormula || '',
          shapeFormula: outputFormula,
          tokenCountFormula: finalBlock?.tokenCountFormula || 'L=1+5+1+5=12,\\ d=6',
          sequenceIndex: lIndex + 1,
          dModelRow: dIndex + 1,
          isFinal: true,
          isOutputMatrix: true,
        };
        interior.add(cube);
        concatCubeTargets.push(cube);
        cubes.push(cube);
        remapCells.push({
          b: 0,
          l: lIndex,
          d: dIndex,
          mesh: cube,
          color: blockColor,
        });

        [1, 2].forEach((layer) => {
          const ghost = new THREE.Mesh(
            new THREE.BoxGeometry(cubeSize * 0.94, cubeSize * 0.94, cubeSize * 0.94),
            new THREE.MeshPhongMaterial({
              color: blockColor,
              transparent: true,
              opacity: Math.max(0.018, 0.10 * (0.38 ** (layer - 1))),
              emissive: new THREE.Color(blockColor),
              emissiveIntensity: 0.035 / layer,
              shininess: 40,
              depthWrite: false,
            }),
          );
          ghost.position.set(x, fixedY - batchStepY * layer, z);
          interior.add(ghost);
          remapCells.push({
            b: layer,
            l: lIndex,
            d: dIndex,
            mesh: ghost,
            color: blockColor,
          });
        });
      }
    }

    const blockCenterX = blockStartX + ((displayTokens - 1) * lStepX) / 2;
    const blockLabel = createOverlaySprite(block.key, {
      width: 132,
      height: 42,
      fontSize: 15,
      scale: 0.085,
      fill: '#dbeafe',
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    blockLabel.position.set(blockCenterX, fixedY + cubeSize * 1.75, matrixFrontZ + cubeSize * 0.2);
    interior.add(blockLabel);

    globalTokenCursor += displayTokens;
  });

  const outputLabel = createFormulaSprite(outputFormula, {
    width: 640,
    height: 54,
    fontSize: 14,
    scale: 0.07,
    align: 'center',
    fill: '#fef3c7',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  outputLabel.position.set(0, fixedY + cubeSize * 3.1, matrixFrontZ + 3);
  interior.add(outputLabel);

  const lAxisLabel = createFormulaSprite('L=1+5+1+5=12', {
    width: 220,
    height: 42,
    fontSize: 13,
    scale: 0.055,
    align: 'center',
    fill: '#dbeafe',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  lAxisLabel.position.set(0, fixedY - cubeSize * 2.9, matrixFrontZ + 2);
  interior.add(lAxisLabel);

  const dAxisLabel = createFormulaSprite('d=6', {
    width: 360,
    height: 42,
    fontSize: 13,
    scale: 0.055,
    align: 'center',
    fill: '#bfdbfe',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  dAxisLabel.position.set(startX - cubeSize * 2.2, fixedY, matrixFrontZ - ((dRows - 1) * dStepZ) / 2);
  interior.add(dAxisLabel);

  const batchHint = createOverlaySprite('B: y-direction stacked copies', {
    width: 280,
    height: 42,
    fontSize: 14,
    scale: 0.08,
    fill: '#bfdbfe',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  batchHint.position.set(0, fixedY - cubeSize * 4.4, matrixFrontZ + 2);
  interior.add(batchHint);

	  group.add(interior);
	  moduleEntry.concatRemapCells = remapCells;
	  moduleAnimators.push((elapsed) => {
    cubes.forEach((cube, index) => {
      const s = 1 + Math.sin(elapsed * 1.65 + index * 0.6) * 0.05;
      cube.scale.setScalar(s);
    });
  });
  moduleEntry.interior = interior;
}

function addOutputEmbeddingVisual(moduleEntry, visualSpec) {
  if (Array.isArray(visualSpec.networks) && visualSpec.networks.length
      && Array.isArray(visualSpec.concatBlocks) && visualSpec.concatBlocks.length) {
    const { group, spec } = moduleEntry;
    const interior = new THREE.Group();
    const activeObjects = [];
    const moduleWidth = spec.size.x || 320;
    const moduleHeight = spec.size.y || 138;
    const fixedZ = 14;
    const networkSpecs = visualSpec.networks;
    const networkCount = networkSpecs.length;
    const spanWidth = moduleWidth * 0.88;
    const networkSpacing = spanWidth / Math.max(1, networkCount);
    const inputCenterY = -moduleHeight * 0.20;
    const outputCenterY = -moduleHeight * 0.01;
    const nodeRadius = 1.75;
    const matrixY = moduleHeight * 0.30;
    const matrixFrontZ = fixedZ + 1;
    const cubeSize = 3.2;
    const dRows = 5;
    const dStepZ = cubeSize * 1.2;
    const batchStepY = -2.5;
    const branchAnchors = [];

    const addText = (text, x, y, {
      width = 86,
      height = 18,
      fontSize = 10,
      fill = '#e2e8f0',
      scale = 0.044,
      z = fixedZ + 8,
    } = {}) => {
      const sprite = createOverlaySprite(text, {
        width,
        height,
        fontSize,
        scale,
        fill,
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      sprite.position.set(x, y, z);
      interior.add(sprite);
      return sprite;
    };

    const addFormula = (formulaText, x, y, {
      width = 160,
      height = 18,
      fontSize = 7.4,
      fill = '#dbeafe',
      scale = 0.028,
      align = 'center',
      z = fixedZ + 8,
    } = {}) => {
      const sprite = createFormulaSprite(formulaText, {
        width,
        height,
        fontSize,
        scale,
        align,
        fill,
        background: 'rgba(0,0,0,0)',
        border: 'rgba(0,0,0,0)',
      });
      sprite.position.set(x, y, z);
      interior.add(sprite);
      return sprite;
    };

    const addLine = (points, color = '#e2e8f0', opacity = 0.28) => {
      const line = createLine(points, color, opacity);
      line.renderOrder = 20;
      interior.add(line);
      return line;
    };

    const addWireBox = ({
      x,
      y,
      z,
      width,
      height,
      depth,
      color = '#f8fafc',
      opacity = 0.28,
    }) => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.01,
          depthWrite: false,
        }),
      );
      mesh.position.set(x, y, z);
      interior.add(mesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity,
          depthWrite: false,
        }),
      );
      edges.position.copy(mesh.position);
      interior.add(edges);
      return {
        center: mesh.position.clone(),
        top: new THREE.Vector3(x, y + height * 0.5, z),
        bottom: new THREE.Vector3(x, y - height * 0.5, z),
      };
    };

    const addTensorGrid = ({
      centerX,
      centerY,
      frontZ,
      batchCount,
      seqCount,
      dCount,
      color,
      label,
      formula,
      xStep = 4.2,
      yStep = 3.6,
      zStep = 5.0,
    }) => {
      const nodes = [];
      for (let b = 0; b < batchCount; b += 1) {
        for (let l = 0; l < seqCount; l += 1) {
          for (let d = 0; d < dCount; d += 1) {
            const node = new THREE.Mesh(
              new THREE.SphereGeometry(nodeRadius, 12, 10),
              new THREE.MeshPhongMaterial({
                color,
                transparent: true,
                opacity: 0.84,
                emissive: new THREE.Color(color),
                emissiveIntensity: 0.10,
                shininess: 74,
              }),
            );
            node.position.set(
              centerX + (d - (dCount - 1) / 2) * xStep,
              centerY + (seqCount === 1 ? 0 : (l - (seqCount - 1) / 2) * yStep),
              frontZ - b * zStep,
            );
            interior.add(node);
            nodes.push(node);
            activeObjects.push(node);
          }
        }
      }

      const xs = nodes.map((mesh) => mesh.position.x);
      const ys = nodes.map((mesh) => mesh.position.y);
      const zs = nodes.map((mesh) => mesh.position.z);
      const frame = addWireBox({
        x: (Math.max(...xs) + Math.min(...xs)) / 2,
        y: (Math.max(...ys) + Math.min(...ys)) / 2,
        z: (Math.max(...zs) + Math.min(...zs)) / 2,
        width: Math.max(...xs) - Math.min(...xs) + nodeRadius * 3.2,
        height: Math.max(...ys) - Math.min(...ys) + nodeRadius * 3.2,
        depth: Math.max(...zs) - Math.min(...zs) + nodeRadius * 3.2,
        color,
        opacity: 0.26,
      });

      addText(label, centerX, frame.top.y + 5.6, {
        width: 104,
        height: 20,
        fontSize: 11.5,
        fill: '#f8fafc',
        scale: 0.05,
      });
      addFormula(formula, centerX, frame.bottom.y - 4.2, {
        width: 300,
        height: 33,
        fontSize: 15.75,
        fill: '#cbd5e1',
        scale: 0.3,
      });
      return { ...frame, nodes };
    };

    networkSpecs.forEach((net, netIdx) => {
      const centerX = -spanWidth * 0.5 + networkSpacing * (netIdx + 0.5);
      const batchCount = Math.max(1, Number(net.batchCount || 3));
      const seqCount = Math.max(1, Number(net.sequenceCount || 1));
      const inputDCount = Math.max(1, Number(net.inputDCount || 4));
      const outputDCount = Math.max(1, Number(net.outputDCount || 5));
      const sourceFormulaLabel = net.sourceId.replace('x_', 'X_{').replace(/$/, '}');
      const batchSymbol = net.batchSymbol || 'B';
      const seqSymbol = net.sequenceLength || (seqCount === 1 ? '1' : 'L');
      const inputDimSymbol = net.inputCount || `d_{${net.key}}`;
      const outputDimSymbol = net.outputCount || 'd_{model}';
      const inputFormula = seqCount === 1
        ? `${sourceFormulaLabel} \\in \\mathbb{R}^{${batchSymbol}\\times ${inputDimSymbol}}`
        : `${sourceFormulaLabel} \\in \\mathbb{R}^{${batchSymbol}\\times ${seqSymbol}\\times ${inputDimSymbol}}`;
      const outputFormula = seqCount === 1
        ? `${net.title} \\in \\mathbb{R}^{${batchSymbol}\\times ${outputDimSymbol}}`
        : `${net.title} \\in \\mathbb{R}^{${batchSymbol}\\times ${seqSymbol}\\times ${outputDimSymbol}}`;
      const weightFormula = `W_{${net.key}} \\in \\mathbb{R}^{${inputDimSymbol}\\times ${outputDimSymbol}}`;

      const inputTensor = addTensorGrid({
        centerX,
        centerY: inputCenterY,
        frontZ: fixedZ + 2,
        batchCount,
        seqCount,
        dCount: inputDCount,
        color: net.color || '#93c5fd',
        label: net.sourceId || `x_${net.key}`,
        formula: inputFormula,
      });
      const outputTensor = addTensorGrid({
        centerX,
        centerY: outputCenterY,
        frontZ: fixedZ + 2,
        batchCount,
        seqCount,
        dCount: outputDCount,
        color: net.color || '#93c5fd',
        label: net.title,
        formula: outputFormula,
      });

      const connectionPositions = [];
      inputTensor.nodes.forEach((fromNode) => {
        outputTensor.nodes.forEach((toNode) => {
          connectionPositions.push(
            fromNode.position.x, fromNode.position.y, fromNode.position.z,
            toNode.position.x, toNode.position.y, toNode.position.z,
          );
        });
      });
      const connectionGeometry = new THREE.BufferGeometry();
      connectionGeometry.setAttribute('position', new THREE.Float32BufferAttribute(connectionPositions, 3));
      const connectionLines = new THREE.LineSegments(
        connectionGeometry,
        new THREE.LineBasicMaterial({
          color: net.color || '#93c5fd',
          transparent: true,
          opacity: 0.10,
          depthWrite: false,
        }),
      );
      connectionLines.renderOrder = 20;
      interior.add(connectionLines);

      addFormula(weightFormula, centerX, (inputTensor.top.y + outputTensor.bottom.y) * 0.5, {
        width: 112,
        height: 16,
        fontSize: 7.0,
        fill: '#fde68a',
        scale: 0.024,
      });

      if (seqCount > 1) {
        addFormula(`x:d=${inputDCount},\\ y:L=${seqCount},\\ z:B=${batchCount}`, centerX, inputTensor.bottom.y - 9.2, {
          width: 152,
          height: 16,
          fontSize: 7.0,
          fill: '#94a3b8',
          scale: 0.025,
        });
      } else {
        addFormula(`x:d=${inputDCount},\\ z:B=${batchCount}`, centerX, inputTensor.bottom.y - 9.2, {
          width: 120,
          height: 16,
          fontSize: 7.0,
          fill: '#94a3b8',
          scale: 0.025,
        });
      }

      branchAnchors.push({
        x: centerX,
        y: outputTensor.top.y + 3.5,
        color: net.color || '#93c5fd',
      });
    });

    const concatYRows = 4;
    const concatZBatches = 3;
    const totalTokenSlots = visualSpec.concatBlocks.reduce((sum, block) => sum + Math.max(1, Number(block.displayTokens || 1)), 0);
    const concatXSpan = moduleWidth * 0.75;
    const concatXStep = totalTokenSlots > 1 ? concatXSpan / (totalTokenSlots - 1) : 0;
    const startX = -concatXSpan * 0.5;
    let globalTokenCursor = 0;
    const blockAnchors = [];
    const concatCells = [];

    visualSpec.concatBlocks.forEach((block) => {
      const displayTokens = Math.max(1, Number(block.displayTokens || 1));
      const blockColor = lerpColorHex(block.color || '#93c5fd', '#ffffff', 0.14);
      const blockStartX = startX + globalTokenCursor * concatXStep;

      for (let lLocal = 0; lLocal < displayTokens; lLocal += 1) {
        const lIndex = globalTokenCursor + lLocal;
        for (let yIndex = 0; yIndex < concatYRows; yIndex += 1) {
          for (let bIndex = 0; bIndex < concatZBatches; bIndex += 1) {
            const cube = new THREE.Mesh(
              new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize),
              new THREE.MeshPhongMaterial({
                color: blockColor,
                transparent: true,
                opacity: 0.54,
                emissive: new THREE.Color(blockColor),
                emissiveIntensity: 0.08,
                shininess: 70,
              }),
            );
            cube.position.set(
              startX + lIndex * concatXStep,
              matrixY + (yIndex - (concatYRows - 1) / 2) * batchStepY * -1.05,
              matrixFrontZ - bIndex * dStepZ * 1.16,
            );
            interior.add(cube);
            activeObjects.push(cube);
            concatCells.push(cube);
          }
        }
      }

      const blockCenterX = blockStartX + ((displayTokens - 1) * concatXStep) * 0.5;
      addText(block.key, blockCenterX, matrixY + 8.6, {
        width: 64,
        height: 16,
        fontSize: 8.5,
        fill: '#dbeafe',
        scale: 0.034,
      });
      blockAnchors.push({
        x: blockCenterX,
        y: matrixY - 7,
        color: block.color || '#93c5fd',
      });
      globalTokenCursor += displayTokens;
    });

    branchAnchors.forEach((anchor, index) => {
      const target = blockAnchors[index] || blockAnchors[blockAnchors.length - 1];
      if (!target) return;
      const elbowY = (anchor.y + target.y) * 0.5;
      addLine([
        new THREE.Vector3(anchor.x, anchor.y, fixedZ + 1),
        new THREE.Vector3(anchor.x, elbowY, fixedZ + 1),
        new THREE.Vector3(target.x, elbowY, fixedZ + 1),
        new THREE.Vector3(target.x, target.y, fixedZ + 1),
      ], anchor.color, 0.24);
    });

    addText('Concat', 0, matrixY + 18, {
      width: 80,
      height: 18,
      fontSize: 10,
      fill: '#fef3c7',
      scale: 0.038,
    });
    addFormula(visualSpec.concatOutputFormula || 'Q_{slot}^{(viz)} \\in \\mathbb{R}^{B\\times L\\times d}', 0, matrixY + 28, {
      width: 220,
      height: 18,
      fontSize: 7.3,
      fill: '#dbeafe',
      scale: 0.026,
    });
    addFormula('x:L=12,\\ y:d=4,\\ z:B=3', 0, matrixY - 17.5, {
      width: 150,
      height: 16,
      fontSize: 7.0,
      fill: '#94a3b8',
      scale: 0.024,
    });

    group.add(interior);
    moduleAnimators.push((elapsed) => {
      activeObjects.forEach((obj, index) => {
        const s = 1 + Math.sin(elapsed * 1.8 + index * 0.26) * 0.035;
        obj.scale.setScalar(s);
      });
      concatCells.forEach((cube, index) => {
        const s = 1 + Math.sin(elapsed * 1.55 + index * 0.22) * 0.04;
        cube.scale.setScalar(s);
      });
    });
    moduleEntry.interior = interior;
    return;
  }

  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const activeObjects = [];
  const fixedZ = 14;
  const moduleWidth = spec.size.x || 220;
  const moduleHeight = spec.size.y || 84;
  const tokenCount = Math.max(1, Number(visualSpec.tokenCount || 6));
  const dRows = Math.max(1, Number(visualSpec.dRows || 6));
  const bLayers = Math.max(1, Number(visualSpec.bLayers || 3));
  const tokenColor = visualSpec.tokenColor || '#fda4af';
  const tensorColor = visualSpec.tensorColor || visualSpec.tint || '#f59e0b';
  const tokenLabels = Array.isArray(visualSpec.tokenLabels) && visualSpec.tokenLabels.length
    ? visualSpec.tokenLabels
    : ['<bos>', 'y_1', 'y_2', 'y_3', '...', 'y_{t-1}'];
  const tokenY = -moduleHeight * 0.2;
  const opY = -moduleHeight * 0.02;
  const tensorY = moduleHeight * 0.16;
  const tokenStepX = (moduleWidth * 0.72) / Math.max(1, tokenCount - 1);
  const tensorXStep = (moduleWidth * 0.68) / Math.max(1, tokenCount - 1);
  const tensorZStep = (moduleHeight * 0.36) / Math.max(1, dRows - 1);
  const tensorYStep = -(moduleHeight * 0.09);
  const tokenWidth = Math.min(22, moduleWidth * 0.12);
  const tokenHeight = Math.min(8, moduleHeight * 0.12);
  const nodeRadius = Math.max(2.1, Math.min(tensorXStep * 0.22, tensorZStep * 0.36));

  const addLabel = (text, x, y, z = fixedZ + 8, {
    width = 110,
    height = 24,
    fontSize = 11,
    fill = '#e2e8f0',
    scale = 0.05,
  } = {}) => {
    const sprite = createOverlaySprite(text, {
      width,
      height,
      fontSize,
      scale,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, z);
    interior.add(sprite);
    return sprite;
  };

  const addFormula = (formulaText, x, y, {
    width = 220,
    height = 26,
    fontSize = 10,
    fill = '#dbeafe',
    scale = 0.038,
    align = 'center',
    z = fixedZ + 8,
  } = {}) => {
    const sprite = createFormulaSprite(formulaText, {
      width,
      height,
      fontSize,
      scale,
      align,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, z);
    interior.add(sprite);
    return sprite;
  };

  const addLink = (points, color = '#e5eefc', opacity = 0.5) => {
    const line = createLine(points, color, opacity);
    line.renderOrder = 20;
    interior.add(line);
    return line;
  };

  const tokenCenters = [];
  for (let idx = 0; idx < tokenCount; idx += 1) {
    const x = (idx - (tokenCount - 1) / 2) * tokenStepX;
    const tokenBox = new THREE.Mesh(
      new THREE.BoxGeometry(tokenWidth, tokenHeight, 5),
      new THREE.MeshPhongMaterial({
        color: tokenColor,
        transparent: true,
        opacity: 0.74,
        emissive: new THREE.Color(tokenColor),
        emissiveIntensity: 0.14,
        shininess: 76,
      }),
    );
    tokenBox.position.set(x, tokenY, fixedZ + 1);
    interior.add(tokenBox);
    activeObjects.push(tokenBox);
    tokenCenters.push(tokenBox.position.clone());

    addLabel(tokenLabels[Math.min(idx, tokenLabels.length - 1)] || `y_${idx}`, x, tokenY - tokenHeight * 1.8, fixedZ + 6, {
      width: 72,
      height: 20,
      fontSize: 10,
      fill: '#fecdd3',
      scale: 0.042,
    });
  }

  const embedBox = new THREE.Mesh(
    new THREE.BoxGeometry(40, 10, 5),
    new THREE.MeshPhongMaterial({
      color: '#fde68a',
      transparent: true,
      opacity: 0.62,
      emissive: new THREE.Color('#fde68a'),
      emissiveIntensity: 0.08,
      shininess: 60,
    }),
  );
  embedBox.position.set(0, opY, fixedZ + 1);
  interior.add(embedBox);
  activeObjects.push(embedBox);
  addLabel('Embed', 0, opY, fixedZ + 6, {
    width: 74,
    height: 22,
    fontSize: 11,
    fill: '#111827',
    scale: 0.042,
  });

  const tensorNodes = [];
  for (let b = 0; b < bLayers; b += 1) {
    for (let l = 0; l < tokenCount; l += 1) {
      for (let d = 0; d < dRows; d += 1) {
        const node = new THREE.Mesh(
          new THREE.SphereGeometry(nodeRadius, 14, 12),
          new THREE.MeshPhongMaterial({
            color: tensorColor,
            transparent: true,
            opacity: 0.84,
            emissive: new THREE.Color(tensorColor),
            emissiveIntensity: 0.12,
            shininess: 84,
          }),
        );
        node.position.set(
          (l - (tokenCount - 1) / 2) * tensorXStep,
          tensorY + b * tensorYStep,
          fixedZ - d * tensorZStep,
        );
        interior.add(node);
        tensorNodes.push(node);
        activeObjects.push(node);
      }
    }
  }

  const xs = tensorNodes.map((mesh) => mesh.position.x);
  const ys = tensorNodes.map((mesh) => mesh.position.y);
  const zs = tensorNodes.map((mesh) => mesh.position.z);
  const tensorFrame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(
      Math.max(...xs) - Math.min(...xs) + nodeRadius * 3,
      Math.max(...ys) - Math.min(...ys) + nodeRadius * 3,
      Math.max(...zs) - Math.min(...zs) + nodeRadius * 3,
    )),
    new THREE.LineBasicMaterial({
      color: tensorColor,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    }),
  );
  tensorFrame.position.set(
    (Math.max(...xs) + Math.min(...xs)) / 2,
    (Math.max(...ys) + Math.min(...ys)) / 2,
    (Math.max(...zs) + Math.min(...zs)) / 2,
  );
  interior.add(tensorFrame);

  tokenCenters.forEach((center) => {
    addLink([
      new THREE.Vector3(center.x, center.y + tokenHeight * 0.65, center.z),
      new THREE.Vector3(center.x, opY - 7.2, center.z),
      new THREE.Vector3(0, opY - 7.2, center.z),
      new THREE.Vector3(0, opY - 5, center.z),
    ], tokenColor, 0.34);
  });
  addLink([
    new THREE.Vector3(0, opY + 5, fixedZ + 1),
    new THREE.Vector3(0, tensorY - 8, fixedZ + 1),
  ], '#fde68a', 0.52);

  addLabel(visualSpec.inputLabel || 'Q_{slot}^{0}', -moduleWidth * 0.28, tokenY + tokenHeight * 2.2, fixedZ + 7, {
    width: 86,
    height: 20,
    fontSize: 11,
    fill: '#fecdd3',
    scale: 0.042,
  });
  addLabel(visualSpec.outputLabel || 'Q_{slot}', 0, tensorY + nodeRadius * 3.6, fixedZ + 8, {
    width: 92,
    height: 22,
    fontSize: 11,
    fill: '#fde68a',
    scale: 0.046,
  });
  addFormula(visualSpec.dimFormula || 'B\\times T_{max}\\times d_{model}', moduleWidth * 0.2, tensorY + nodeRadius * 5.1, {
    width: 180,
    height: 22,
    fontSize: 9,
    fill: '#dbeafe',
    scale: 0.032,
    align: 'right',
  });

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    activeObjects.forEach((obj, index) => {
      const s = 1 + Math.sin(elapsed * 1.9 + index * 0.28) * 0.035;
      obj.scale.setScalar(s);
    });
  });
  moduleEntry.interior = interior;
}

function addAddNormVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const activeObjects = [];
  const fixedZ = 14;

  const cubeMaterial = (rawColor, opacity = 0.68) => new THREE.MeshPhongMaterial({
    color: rawColor,
    transparent: true,
    opacity,
    emissive: new THREE.Color(rawColor),
    emissiveIntensity: 0.08,
    shininess: 64,
  });

  const addLabel = (text, x, y, z = fixedZ + 9, {
    width = 120,
    height = 26,
    fontSize = 12,
    fill = '#e2e8f0',
    scale = 0.05,
  } = {}) => {
    const sprite = createOverlaySprite(text, {
      width,
      height,
      fontSize,
      scale,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, z);
    interior.add(sprite);
    return sprite;
  };

  const addFormula = (formulaText, x, y, {
    width = 240,
    height = 28,
    fontSize = 10,
    fill = '#dbeafe',
    scale = 0.038,
    align = 'center',
    z = fixedZ + 9,
  } = {}) => {
    const sprite = createFormulaSprite(formulaText, {
      width,
      height,
      fontSize,
      scale,
      align,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, z);
    interior.add(sprite);
    return sprite;
  };

  const addWireBox = ({
    x,
    y,
    z,
    width,
    height,
    depth,
    color = '#fef3c7',
    opacity = 0.44,
  }) => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, depth),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.02,
        depthWrite: false,
      }),
    );
    mesh.position.set(x, y, z);
    interior.add(mesh);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      }),
    );
    edges.position.copy(mesh.position);
    interior.add(edges);
    return {
      center: mesh.position.clone(),
      top: new THREE.Vector3(x, y + height * 0.5, z),
      bottom: new THREE.Vector3(x, y - height * 0.5, z),
      left: new THREE.Vector3(x - width * 0.5, y, z),
      right: new THREE.Vector3(x + width * 0.5, y, z),
    };
  };

  const addMiniTensor = ({
    label,
    formula,
    x,
    y,
    color,
    segmentColors = null,
    nodeShape = 'box',
    lCols = 12,
    dRows = 6,
    bLayers = 3,
    cell = 1.1,
    xStep = 2.45,
    zStep = 1.6,
    yStep = -1.65,
    opacity = 0.62,
    labelFill = '#dbeafe',
    frameOpacity = 0.34,
    showLabel = true,
    showFormula = true,
    labelYOffset = 5,
    formulaYOffset = -4.4,
  }) => {
    const nodes = [];
    for (let b = 0; b < bLayers; b += 1) {
      for (let l = 0; l < lCols; l += 1) {
        for (let d = 0; d < dRows; d += 1) {
          const nodeColor = Array.isArray(segmentColors) && segmentColors[l] ? segmentColors[l] : color;
          const geometry = nodeShape === 'sphere'
            ? new THREE.SphereGeometry(cell * 0.5, 12, 10)
            : new THREE.BoxGeometry(cell, cell, cell);
          const node = new THREE.Mesh(geometry, cubeMaterial(nodeColor, opacity));
          node.position.set(
            x + (l - (lCols - 1) / 2) * xStep,
            y + b * yStep,
            fixedZ - d * zStep,
          );
          interior.add(node);
          nodes.push(node);
          activeObjects.push(node);
        }
      }
    }
    const xs = nodes.map((mesh) => mesh.position.x);
    const ys = nodes.map((mesh) => mesh.position.y);
    const zs = nodes.map((mesh) => mesh.position.z);
    const width = Math.max(...xs) - Math.min(...xs) + cell * 1.8;
    const height = Math.max(...ys) - Math.min(...ys) + cell * 1.8;
    const depth = Math.max(...zs) - Math.min(...zs) + cell * 1.8;
    const centerX = (Math.max(...xs) + Math.min(...xs)) / 2;
    const centerY = (Math.max(...ys) + Math.min(...ys)) / 2;
    const centerZ = (Math.max(...zs) + Math.min(...zs)) / 2;
    const frame = addWireBox({
      x: centerX,
      y: centerY,
      z: centerZ,
      width,
      height,
      depth,
      color: color || '#fef3c7',
      opacity: frameOpacity,
    });
    if (showLabel) {
      addLabel(label, centerX, frame.top.y + labelYOffset, centerZ + depth * 0.5 + 1.4, {
        width: 110,
        height: 24,
        fontSize: 11,
        fill: labelFill,
        scale: 0.046,
      });
    }
    if (showFormula) {
      addFormula(formula, centerX, frame.bottom.y + formulaYOffset, {
        width: 190,
        height: 22,
        fontSize: 9,
        fill: '#cbd5e1',
        scale: 0.032,
      });
    }
    return { ...frame, width, height, depth };
  };

  const addOpBox = (label, x, y, {
    width = 22,
    height = 10,
    color = '#fde68a',
    fill = '#111827',
  } = {}) => {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, 5),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: 0.58,
        emissive: new THREE.Color(color),
        emissiveIntensity: 0.08,
        shininess: 60,
      }),
    );
    box.position.set(x, y, fixedZ + 1);
    interior.add(box);
    addLabel(label, x, y, fixedZ + 6, {
      width: 90,
      height: 22,
      fontSize: 11,
      fill,
      scale: 0.044,
    });
    return {
      center: box.position.clone(),
      top: new THREE.Vector3(x, y + height * 0.56, fixedZ + 1),
      bottom: new THREE.Vector3(x, y - height * 0.56, fixedZ + 1),
      left: new THREE.Vector3(x - width * 0.56, y, fixedZ + 1),
      right: new THREE.Vector3(x + width * 0.56, y, fixedZ + 1),
    };
  };

  const addLink = (points, color = '#e5eefc', opacity = 0.52) => {
    const line = createLine(points, color, opacity);
    line.renderOrder = 20;
    interior.add(line);
    return line;
  };
  const addAddNode = (x, y, {
    radius = 4.2,
    color = '#fde68a',
    label = '+',
  } = {}) => {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 20, 18),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: 0.82,
        emissive: new THREE.Color(color),
        emissiveIntensity: 0.14,
        shininess: 78,
      }),
    );
    mesh.position.set(x, y, fixedZ + 1.2);
    interior.add(mesh);
    activeObjects.push(mesh);
    addLabel(label, x, y, fixedZ + 6.8, {
      width: 28,
      height: 22,
      fontSize: 16,
      fill: '#111827',
      scale: 0.042,
    });
    return {
      center: mesh.position.clone(),
      top: new THREE.Vector3(x, y + radius, fixedZ + 1.2),
      bottom: new THREE.Vector3(x, y - radius, fixedZ + 1.2),
      left: new THREE.Vector3(x - radius, y, fixedZ + 1.2),
      right: new THREE.Vector3(x + radius, y, fixedZ + 1.2),
      radius,
    };
  };

  const moduleWidth = spec?.size?.x || 250;
  const moduleHeight = spec?.size?.y || 74;
  const visualL = Math.max(1, Number(visualSpec.lCols || 12));
  const visualD = Math.max(1, Number(visualSpec.dRows || 6));
  const visualB = Math.max(1, Number(visualSpec.bLayers || 3));
  const inputY = -moduleHeight * 0.12;
  const leftInputX = -moduleWidth * 0.24;
  const rightInputX = moduleWidth * 0.24;
  const addNodeY = moduleHeight * 0.05;
  const normY = moduleHeight * 0.15;
  const outputY = moduleHeight * 0.42;

  const defaultConcatSegments = [
    '#61dafb',
    '#fbbf24', '#fbbf24', '#fbbf24', '#fbbf24', '#fbbf24',
    '#7ee787',
    '#a78bfa', '#a78bfa', '#a78bfa', '#a78bfa', '#a78bfa',
  ];
  const tensorShapeDefaults = {
    box: {
      cell: 7.2 * 0.85,
      xStep: (7.2 * 0.85) * 1.38,
      zStep: (7.2 * 0.85) * 1.38,
      yStep: -((7.2 * 0.85) * 1.28),
      opacity: 0.62,
    },
    sphere: {
      cell: 2.85 * 1.08,
      xStep: 7.2 * 1.38,
      zStep: 2.85 * 1.18,
      yStep: -3.4,
      opacity: 0.62,
    },
  };
  const concatNodeShape = visualSpec.concatNodeShape || visualSpec.leftNodeShape || 'box';
  const attentionNodeShape = visualSpec.attentionNodeShape || visualSpec.rightNodeShape || 'sphere';
  const concatTensorDefaults = tensorShapeDefaults[concatNodeShape] || tensorShapeDefaults.box;
  const attentionTensorDefaults = tensorShapeDefaults[attentionNodeShape] || tensorShapeDefaults.sphere;
  const concatTensor = addMiniTensor({
    label: visualSpec.concatLabel || 'X_{enc}',
    formula: 'B\\times L\\times d',
    x: leftInputX,
    y: inputY,
    color: visualSpec.concatColor || '#93c5fd',
    lCols: visualL,
    dRows: visualD,
    bLayers: visualB,
    segmentColors: concatNodeShape === 'box' ? (visualSpec.segmentColors || defaultConcatSegments) : null,
    nodeShape: concatNodeShape,
    ...concatTensorDefaults,
    labelFill: visualSpec.concatLabelFill || (concatNodeShape === 'box' ? '#bfdbfe' : '#fef9c3'),
    showFormula: false,
    frameOpacity: 0.3,
  });
  const attentionTensor = addMiniTensor({
    label: visualSpec.attentionLabel || 'H_{out}',
    formula: 'B\\times L\\times d',
    x: rightInputX,
    y: inputY,
    color: visualSpec.attentionColor || '#38bdf8',
    lCols: visualL,
    dRows: visualD,
    bLayers: visualB,
    nodeShape: attentionNodeShape,
    ...attentionTensorDefaults,
    labelFill: visualSpec.attentionLabelFill || '#bae6fd',
    showFormula: false,
    frameOpacity: 0.3,
  });
  const addNode = addAddNode(0, addNodeY, {
    radius: 4.8,
    color: '#fde68a',
    label: '+',
  });
  const normBox = addOpBox('Norm', 0, normY, { width: 40, height: 9, color: '#fef3c7' });
  const outputTensor = addMiniTensor({
    label: visualSpec.outputLabel || 'H^{norm}',
    formula: 'B\\times L\\times d',
    x: 2,
    y: outputY,
    color: visualSpec.outputColor || '#fef3c7',
    lCols: visualL,
    dRows: visualD,
    bLayers: visualB,
    nodeShape: 'sphere',
    cell: 2.85 * 1.08,
    xStep: 7.2 * 1.38,
    zStep: 2.85 * 1.18,
    yStep: -3.4,
    labelFill: '#fef9c3',
    showFormula: false,
    frameOpacity: 0.34,
    labelYOffset: 3.2,
  });
  addLink([
    new THREE.Vector3(concatTensor.center.x, concatTensor.top.y, concatTensor.center.z),
    new THREE.Vector3(concatTensor.center.x, addNode.bottom.y - 2.6, concatTensor.center.z),
    new THREE.Vector3(addNode.left.x - 2.4, addNode.bottom.y - 2.6, addNode.center.z),
    new THREE.Vector3(addNode.left.x, addNode.center.y, addNode.center.z),
  ], '#93c5fd', 0.58);
  addLink([
    new THREE.Vector3(attentionTensor.center.x, attentionTensor.top.y, attentionTensor.center.z),
    new THREE.Vector3(attentionTensor.center.x, addNode.bottom.y - 2.6, attentionTensor.center.z),
    new THREE.Vector3(addNode.right.x + 2.4, addNode.bottom.y - 2.6, addNode.center.z),
    new THREE.Vector3(addNode.right.x, addNode.center.y, addNode.center.z),
  ], '#38bdf8', 0.58);
  addLink([
    new THREE.Vector3(addNode.center.x, addNode.top.y, addNode.center.z),
    new THREE.Vector3(addNode.center.x, normBox.bottom.y, addNode.center.z),
  ], '#fde68a', 0.56);
  addLink([
    new THREE.Vector3(normBox.top.x, normBox.top.y, normBox.center.z),
    new THREE.Vector3(outputTensor.center.x, outputTensor.bottom.y, outputTensor.center.z),
  ], '#fef3c7', 0.56);

  addFormula(visualSpec.addFormulaText || 'Z_{b,l,d}=X_{enc,b,l,d}+H_{out,b,l,d}', 0, addNode.bottom.y - 5.8, {
    width: 250,
    height: 22,
    fontSize: 9,
    fill: '#fde68a',
    scale: 0.034,
  });
  addFormula(visualSpec.normFormulaText || 'H^{norm}=LayerNorm(Z)', 0, normBox.top.y + 4.8, {
    width: 182,
    height: 22,
    fontSize: 9,
    fill: '#fef9c3',
    scale: 0.034,
  });
  addFormula(visualSpec.outputFormulaText || 'B\\times L\\times d', moduleWidth * 0.35, outputTensor.top.y + 3.8, {
    width: 128,
    height: 20,
    fontSize: 9,
    fill: '#fef9c3',
    scale: 0.033,
    align: 'right',
  });

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    activeObjects.forEach((obj, index) => {
      const s = 1 + Math.sin(elapsed * 1.8 + index * 0.24) * 0.03;
      obj.scale.setScalar(s);
    });
  });
  moduleEntry.interior = interior;
}

function addFeedForwardVisual(moduleEntry, visualSpec) {
  const { group, spec } = moduleEntry;
  const interior = new THREE.Group();
  const accent = visualSpec.color || visualSpec.tint || spec.haloColor || '#93c5fd';
  const inputColor = visualSpec.inputColor || '#fef3c7';
  const outputColor = visualSpec.outputColor || accent;
  const moduleWidth = spec.size.x || 226;
  const moduleHeight = spec.size.y || 56;
  const centerZ = 12;
  const lCols = Math.max(1, Number(visualSpec.lCols || 12));
  const dRows = Math.max(1, Number(visualSpec.dRows || 6));
  const bLayers = Math.max(1, Number(visualSpec.bLayers || 3));
  const xStep = (moduleWidth * 0.74) / Math.max(1, lCols - 1);
  const zStep = (moduleHeight * 0.62) / Math.max(1, dRows - 1);
  const yStep = -(moduleHeight * 0.09);
  const nodeRadius = Math.max(2.4, Math.min(xStep * 0.32, zStep * 0.42, Math.abs(yStep) * 0.62));
  const inputY = -moduleHeight * 0.18;
  const outputY = moduleHeight * 0.18;
  const inputNodes = [];
  const outputNodes = [];
  const activeObjects = [];

  const addFormulaLocal = (formulaText, x, y, {
    width = 180,
    height = 24,
    fontSize = 10,
    fill = '#dbeafe',
    scale = 0.034,
    align = 'left',
  } = {}) => {
    const sprite = createFormulaSprite(formulaText, {
      width,
      height,
      fontSize,
      scale,
      align,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, centerZ + 8);
    interior.add(sprite);
    return sprite;
  };

  const addTextLocal = (text, x, y, {
    width = 100,
    height = 22,
    fontSize = 11,
    fill = '#e2e8f0',
    scale = 0.042,
  } = {}) => {
    const sprite = createOverlaySprite(text, {
      width,
      height,
      fontSize,
      scale,
      fill,
      background: 'rgba(0,0,0,0)',
      border: 'rgba(0,0,0,0)',
    });
    sprite.position.set(x, y, centerZ + 8);
    interior.add(sprite);
    return sprite;
  };

  const addTensorLayer = (baseY, bucket, label, tensorColor, labelFill) => {
    for (let b = 0; b < bLayers; b += 1) {
      for (let l = 0; l < lCols; l += 1) {
        for (let d = 0; d < dRows; d += 1) {
          const node = new THREE.Mesh(
            new THREE.SphereGeometry(nodeRadius, 16, 12),
            new THREE.MeshPhongMaterial({
              color: tensorColor,
              transparent: true,
              opacity: 0.88,
              emissive: new THREE.Color(tensorColor),
              emissiveIntensity: 0.14,
              shininess: 86,
            }),
          );
          node.position.set(
            (l - (lCols - 1) / 2) * xStep,
            baseY + b * yStep,
            centerZ - d * zStep,
          );
          interior.add(node);
          bucket.push(node);
          activeObjects.push(node);
        }
      }
    }
    addTextLocal(label, -moduleWidth * 0.27, baseY + (label === (visualSpec.inputLabel || 'H^{norm}') ? -nodeRadius * 2.5 : nodeRadius * 2.9), {
      width: 92,
      height: 22,
      fontSize: 11,
      fill: labelFill,
      scale: 0.04,
    });
  };

  addTensorLayer(inputY, inputNodes, visualSpec.inputLabel || 'H^{norm}', inputColor, '#fef9c3');
  addTensorLayer(outputY, outputNodes, visualSpec.outputLabel || 'FFN(H)', outputColor, '#bfdbfe');

  const connectionPositions = [];
  inputNodes.forEach((fromNode) => {
    outputNodes.forEach((toNode) => {
      connectionPositions.push(
        fromNode.position.x, fromNode.position.y, fromNode.position.z,
        toNode.position.x, toNode.position.y, toNode.position.z,
      );
    });
  });
  const connectionGeometry = new THREE.BufferGeometry();
  connectionGeometry.setAttribute('position', new THREE.Float32BufferAttribute(connectionPositions, 3));
  const connectionLines = new THREE.LineSegments(
    connectionGeometry,
    new THREE.LineBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.08,
      depthWrite: false,
    }),
  );
  connectionLines.renderOrder = 18;
  interior.add(connectionLines);

  addTextLocal('Linear', 0, 0, {
    width: 80,
    height: 20,
    fontSize: 10,
    fill: '#f8fafc',
    scale: 0.038,
  });
  addFormulaLocal(visualSpec.dimFormula || 'B\\times L\\times d', spec.size.x * 0.25, outputY + nodeRadius * 2.6, {
    width: 136,
    height: 22,
    fontSize: 9,
    fill: '#dbeafe',
    scale: 0.032,
    align: 'right',
  });
  addFormulaLocal(visualSpec.dimFormula || 'B\\times L\\times d', -spec.size.x * 0.26, inputY - nodeRadius * 2.5, {
    width: 136,
    height: 22,
    fontSize: 9,
    fill: '#cbd5e1',
    scale: 0.032,
    align: 'left',
  });

  group.add(interior);
  moduleAnimators.push((elapsed) => {
    activeObjects.forEach((node, index) => {
      const s = 1 + Math.sin(elapsed * 1.8 + index * 0.4) * 0.035;
      node.scale.setScalar(s);
    });
  });
  moduleEntry.interior = interior;
}

function addModuleInteriorVisual(moduleEntry) {
  const visualSpec = MODULE_INTERIOR_SPECS[moduleEntry.id];
  if (!visualSpec) return;
  if (visualSpec.kind === 'multiheadAttention') {
    addMultiHeadAttentionVisual(moduleEntry, visualSpec);
    return;
  }
  if (visualSpec.kind === 'outputEmbedding') {
    addOutputEmbeddingVisual(moduleEntry, visualSpec);
    return;
  }
  if (visualSpec.kind === 'feedForward') {
    addFeedForwardVisual(moduleEntry, visualSpec);
    return;
  }
  if (visualSpec.kind === 'addNorm') {
    addAddNormVisual(moduleEntry, visualSpec);
    return;
  }
  if (moduleEntry.id === 'inputEmbedding') {
    addEmbeddingNetworkVisual(moduleEntry, visualSpec);
    return;
  }
  if (moduleEntry.id === 'concatPanel') {
    addConcatModuleVisual(moduleEntry, visualSpec);
    return;
  }
  addVariableModuleVisual(moduleEntry, visualSpec);
}

function getModuleWorldPoint(id, localX = 0, localY = 0, localZ = 8) {
  const module = moduleMap.get(id);
  if (!module) return new THREE.Vector3();
  return module.group.localToWorld(new THREE.Vector3(localX, localY, localZ));
}

function getPaperWorldPoint(paper, localX = 0, localY = 0, localZ = 0) {
  return paper.group.localToWorld(new THREE.Vector3(localX, localY, localZ));
}

function getModuleTopCenter(id, lift = 0, z = 8) {
  const module = moduleMap.get(id);
  if (!module) return new THREE.Vector3();
  return getModuleWorldPoint(id, 0, module.spec.size.y / 2 + lift, z);
}

function getModuleBottomCenter(id, drop = 0, z = 8) {
  const module = moduleMap.get(id);
  if (!module) return new THREE.Vector3();
  return getModuleWorldPoint(id, 0, -module.spec.size.y / 2 - drop, z);
}

function getPaperTopCenter(paper, lift = 0) {
  return getPaperWorldPoint(paper, 0, paper.boardHeight * 0.5 + lift, paper.boardDepth * 0.5);
}

function buildTopToBottomArrowPoints(start, end, {
  sourceLift = 26,
  targetDrop = 18,
} = {}) {
  // Keep path in one z-plane so shaft and arrowhead stay coplanar after extrusion.
  const planeZ = Math.max(start.z, end.z);
  const startP = new THREE.Vector3(start.x, start.y, planeZ);
  const endP = new THREE.Vector3(end.x, end.y, planeZ);
  const raisedStart = startP.clone().add(new THREE.Vector3(0, sourceLift, 0));
  const loweredEnd = endP.clone().add(new THREE.Vector3(0, -targetDrop, 0));
  if (Math.abs(startP.x - endP.x) < 6) {
    return [startP, endP];
  }
  return [
    startP,
    raisedStart,
    new THREE.Vector3(endP.x, raisedStart.y, planeZ),
    loweredEnd,
    endP,
  ];
}

function createArrowHeadGeometry(width = 13, length = 18, depth = 4) {
  const shape = new THREE.Shape();
  shape.moveTo(0, length * 0.5);
  shape.lineTo(width * 0.5, -length * 0.5);
  shape.lineTo(-width * 0.5, -length * 0.5);
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: false,
  });
  geometry.translate(0, 0, -depth * 0.5);
  return geometry;
}

function createLiveArrow(getPoints, {
  color = '#f8fafc',
  opacity = 0.78,
  shaftWidth = 6,
  headWidth = 13,
  headLength = 18,
  depth = 4,
  maxSegments = 6,
} = {}) {
  const group = new THREE.Group();
  group.renderOrder = 18;
  webglScene.add(group);

  const shaftGeometry = new THREE.BoxGeometry(shaftWidth, 1, depth);
  const shaftMaterial = new THREE.MeshPhongMaterial({
    color,
    transparent: true,
    opacity,
    shininess: 52,
    depthTest: false,
    depthWrite: false,
  });
  const shaftMeshes = Array.from({ length: maxSegments }, () => {
    const mesh = new THREE.Mesh(shaftGeometry, shaftMaterial.clone());
    mesh.visible = false;
    mesh.renderOrder = 18;
    group.add(mesh);
    return mesh;
  });

  const head = new THREE.Mesh(
    createArrowHeadGeometry(headWidth, headLength, depth + 0.6),
    new THREE.MeshPhongMaterial({
      color,
      transparent: true,
      opacity: Math.min(1, opacity + 0.14),
      shininess: 62,
      depthTest: false,
      depthWrite: false,
    }),
  );
  head.visible = false;
  head.renderOrder = 19;
  group.add(head);

  liveArrows.push({
    getPoints,
    group,
    shaftMeshes,
    head,
    headLength,
  });
}

function updateLiveArrows() {
  liveArrows.forEach((arrow) => {
    const points = arrow.getPoints().filter(Boolean);
    arrow.shaftMeshes.forEach((mesh) => {
      mesh.visible = false;
    });
    arrow.head.visible = false;
    if (points.length < 2) return;

    const segmentCount = points.length - 1;
    for (let i = 0; i < segmentCount; i += 1) {
      const start = points[i];
      const end = points[i + 1];
      const vector = end.clone().sub(start);
      const length = vector.length();
      if (length < 0.6 || i >= arrow.shaftMeshes.length) continue;
      const direction = vector.clone().normalize();
      const isLast = i === segmentCount - 1;
      const shaftLength = Math.max(0.4, isLast ? (length - arrow.headLength * 0.78) : length);
      const shaft = arrow.shaftMeshes[i];
      shaft.visible = true;
      shaft.scale.set(1, shaftLength, 1);
      const center = start.clone().add(direction.clone().multiplyScalar(shaftLength * 0.5));
      shaft.position.copy(center);
      shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

      if (isLast) {
        const tip = end.clone();
        const headCenter = tip.clone().sub(direction.clone().multiplyScalar(arrow.headLength * 0.5));
        arrow.head.visible = true;
        arrow.head.position.copy(headCenter);
        arrow.head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      }
    }
  });
}

function clearSelectedEmbeddingNode() {
  if (!selectedEmbeddingNode || !selectedEmbeddingNode.mesh) return;
  selectedEmbeddingNode.mesh.material.emissiveIntensity = selectedEmbeddingNode.baseEmissive;
  selectedEmbeddingNode.mesh.scale.copy(selectedEmbeddingNode.baseScale);
  selectedEmbeddingNode = null;
}

function getEmbeddingNodeDisplayIndex(info) {
  const fallbackNodeIndex = info?.neuronDisplayIndex ?? (info?.neuronIndex + 1);
  if (info?.matrixCoord) {
    const { row, col, isLastRow, isLastCol, sequenceLength, inputDim } = info.matrixCoord;
    if (isLastRow && isLastCol) return `${sequenceLength}, ${inputDim}`;
    if (isLastRow) return `${inputDim}, col ${col}`;
    if (isLastCol) return `row ${row}, ${sequenceLength}`;
    return `row ${row}, col ${col}`;
  }
  if (info?.isLastVisibleNode) {
    // Last visible node represents the current layer dimensionality symbol/value.
    return info.layerDimensionDisplay || fallbackNodeIndex;
  }
  return fallbackNodeIndex;
}

function buildEmbeddingNodeDimensionInfo(info, { stage = '神经元参数查看', includeWeights = false } = {}) {
  const netSource = String(info.networkSource || info.networkKey || 'x_*').toUpperCase();
  const nodeIndex = getEmbeddingNodeDisplayIndex(info);
  const upperCountText = info.upperWeightCount == null ? '无（输入层）' : `${info.upperWeightCount}`;
  const upperDimText = info.upperWeightFormula || '无（输入层）';
  const lowerDimText = info.lowerWeightFormula || '无（输出层）';
  const lowerCountText = info.lowerWeightCount == null ? '无（输出层）' : `${info.lowerWeightCount}`;
  const branchOutputText = info.branchOutputShape || info.outputFormula || 'E_* \\in \\mathbb{R}^{B\\times L\\times d_{model}}';
  const matrixText = info.matrixCoord
    ? `\n矩阵位置：row ${info.matrixCoord.row}/${info.matrixCoord.rows}, col ${info.matrixCoord.col}/${info.matrixCoord.cols}\n边界含义：最后一列表示 ${info.matrixCoord.sequenceLength}，最后一行表示 ${info.matrixCoord.inputDim}；后方等距层叠表示 batch B。`
    : '';
  const weightPreview = includeWeights
    ? `\n示例权重值：\n${info.weights.map((w, idx) => `w_${idx + 1}=${w.toFixed(3)}`).join(', ')}`
    : '';
  return {
    module: `${netSource}, Layer ${info.layerIndex + 1}, Node ${nodeIndex}`,
    stage,
    before: `上层网络权重数量：\n${upperCountText}\n上层权重维度：\n${upperDimText}`,
    after: `当前节点维度：\n${info.currentNodeFormula}\n下层连接数量：\n${lowerCountText}`,
    detail: `支路输出形状：\n${branchOutputText}${matrixText}\n下层权重维度：\n${lowerDimText}${weightPreview}`,
  };
}

function applyEmbeddingNodeDimensionInfo(info, { stage = '神经元参数查看', includeWeights = false } = {}) {
  dashboardState.focusId = 'inputEmbedding';
  dashboardState.dimensionInfo = buildEmbeddingNodeDimensionInfo(info, { stage, includeWeights });
  updateDashboard();
}

function setEmbeddingTooltipContent(info) {
  if (!info) {
    embeddingTooltip.style.display = 'none';
    return;
  }
  const netSource = String(info.networkSource || info.networkKey || 'x_*').toUpperCase();
  const nodeIndex = getEmbeddingNodeDisplayIndex(info);
  const upperCountText = info.upperWeightCount == null ? '无（输入层）' : String(info.upperWeightCount);
  const upperDimText = info.upperWeightFormula || '无（输入层）';
  const lowerDimText = info.lowerWeightFormula || '无（输出层）';
  const branchOutputText = info.branchOutputShape || info.outputFormula || 'E_* \\in \\mathbb{R}^{B\\times L\\times d_{model}}';
  const matrixHtml = info.matrixCoord
    ? `<div>矩阵位置: <span style="color:#bfdbfe">row ${info.matrixCoord.row}/${info.matrixCoord.rows}, col ${info.matrixCoord.col}/${info.matrixCoord.cols}</span></div>`
    : '';
  const currentDimHtml = formulaTextToHtml(info.currentNodeFormula || '');
  const upperCountHtml = info.upperWeightCount == null ? upperCountText : formulaTextToHtml(upperCountText);
  const upperDimHtml = info.upperWeightFormula ? formulaTextToHtml(upperDimText) : upperDimText;
  const lowerDimHtml = info.lowerWeightFormula ? formulaTextToHtml(lowerDimText) : lowerDimText;
  const branchOutputHtml = formulaTextToHtml(branchOutputText);
  const headerHtml = formulaTextToHtml(`${netSource}, Layer ${info.layerIndex + 1}, Node ${nodeIndex}`);
  embeddingTooltip.innerHTML = [
    `<div style="color:#fbbf24;font-weight:700;margin-bottom:3px;">${headerHtml}</div>`,
    `<div>上层权重数量: <span style="color:#93c5fd">${upperCountHtml}</span></div>`,
    `<div>当前节点维度: <span style="color:#a7f3d0">${currentDimHtml}</span></div>`,
    matrixHtml,
    `<div>支路输出形状: <span style="color:#fde68a">${branchOutputHtml}</span></div>`,
    `<div>下层权重维度: <span style="color:#fda4af">${lowerDimHtml}</span></div>`,
    `<div style="margin-top:3px;color:#cbd5e1;">上层权重维度: ${upperDimHtml}</div>`,
  ].join('');
  embeddingTooltip.style.display = 'block';
}

function moveEmbeddingTooltip(event) {
  if (!event) return;
  embeddingTooltip.style.left = `${event.clientX + 14}px`;
  embeddingTooltip.style.top = `${event.clientY - 24}px`;
}

function clearHoveredEmbeddingNode() {
  if (hoveredEmbeddingNode && hoveredEmbeddingNode.mesh) {
    const isSelected = selectedEmbeddingNode && selectedEmbeddingNode.mesh === hoveredEmbeddingNode.mesh;
    if (!isSelected) {
      hoveredEmbeddingNode.mesh.material.emissiveIntensity = hoveredEmbeddingNode.baseEmissive;
      hoveredEmbeddingNode.mesh.scale.copy(hoveredEmbeddingNode.baseScale);
    }
  }
  hoveredEmbeddingNode = null;
  embeddingTooltip.style.display = 'none';
  if (selectedEmbeddingNode && selectedEmbeddingNode.mesh && selectedEmbeddingNode.mesh.userData?.embeddingNode) {
    applyEmbeddingNodeDimensionInfo(selectedEmbeddingNode.mesh.userData.embeddingNode, {
      stage: '神经元参数查看',
      includeWeights: true,
    });
    return;
  }
  if (dashboardState.focusId === 'inputEmbedding') {
    setDimensionState('inputEmbedding', 'default');
  }
}

function handleEmbeddingNodeHover(nodeMesh) {
  if (!nodeMesh || !nodeMesh.userData || !nodeMesh.userData.embeddingNode) return;
  if (hoveredEmbeddingNode && hoveredEmbeddingNode.mesh === nodeMesh) return;
  clearHoveredEmbeddingNode();
  hoveredEmbeddingNode = {
    mesh: nodeMesh,
    baseEmissive: nodeMesh.material.emissiveIntensity || 0,
    baseScale: nodeMesh.scale.clone(),
  };
  const isSelected = selectedEmbeddingNode && selectedEmbeddingNode.mesh === nodeMesh;
  if (!isSelected) {
    nodeMesh.material.emissiveIntensity = (nodeMesh.material.emissiveIntensity || 0) + 0.26;
    nodeMesh.scale.set(1.1, 1.1, 1.1);
  }
  const info = nodeMesh.userData.embeddingNode;
  const netSource = String(info.networkSource || info.networkKey || 'x_*').toUpperCase();
  const nodeIndex = getEmbeddingNodeDisplayIndex(info);
  setStatus(
    '神经元悬停',
    `悬停在 ${netSource} 支路第 ${info.layerIndex + 1} 层神经元 ${nodeIndex}，已显示上下层权重和当前节点维度。`,
  );
  setEmbeddingTooltipContent(info);
  applyEmbeddingNodeDimensionInfo(info, { stage: '神经元悬停', includeWeights: false });
}

function handleEmbeddingNodeSelection(nodeMesh) {
  if (!nodeMesh || !nodeMesh.userData || !nodeMesh.userData.embeddingNode) return;
  clearSelectedEmbeddingNode();
  const info = nodeMesh.userData.embeddingNode;
  const netSource = String(info.networkSource || info.networkKey || 'x_*').toUpperCase();
  const nodeIndex = getEmbeddingNodeDisplayIndex(info);
  hoveredEmbeddingNode = nodeMesh;
  selectedEmbeddingNode = {
    mesh: nodeMesh,
    baseEmissive: nodeMesh.material.emissiveIntensity || 0,
    baseScale: nodeMesh.scale.clone(),
  };
  nodeMesh.material.emissiveIntensity = (nodeMesh.material.emissiveIntensity || 0) + 0.42;
  nodeMesh.scale.set(1.2, 1.2, 1.2);
  setModuleActive('inputEmbedding', true);
  setDiagramSelection('inputEmbedding');
  setActiveModuleButton('inputEmbedding');
  setStatus(
    '神经元详情',
    `已选中 ${netSource} 支路第 ${info.layerIndex + 1} 层神经元 ${nodeIndex}。右侧面板已刷新维度和权重信息。`,
  );
  applyEmbeddingNodeDimensionInfo(info, { stage: '神经元参数查看', includeWeights: true });
}

function getWebglIntersections(event) {
  const rect = webglRenderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const neuronHit = raycaster
    .intersectObjects(embeddingNeuronTargets, false)
    .find((item) => item.object && item.object.userData && item.object.userData.embeddingNode);
  const concatCubeHit = raycaster
    .intersectObjects(concatCubeTargets, false)
    .find((item) => item.object && item.object.userData && item.object.userData.concatCube);
  const attentionTensorHit = raycaster
    .intersectObjects(attentionTensorTargets, false)
    .find((item) => item.object && item.object.userData && item.object.userData.attentionTensorNode);
  const moduleHit = raycaster
    .intersectObjects(moduleClickTargets, false)
    .find((item) => item.object && item.object.userData && item.object.userData.boxId);
  return { neuronHit, concatCubeHit, attentionTensorHit, moduleHit };
}

function createPlainLabel(text, {
  width = 360,
  height = 120,
  fontSize = 42,
  fill = '#101010',
} = {}) {
  return createTextSprite(text, {
    width,
    height,
    fontSize,
    scale: 0.28,
    fill,
    background: 'rgba(255,255,255,0.0)',
    border: 'rgba(255,255,255,0.0)',
  });
}

function getDimensionState(id, stateKey = 'default') {
  const stateGroup = DIMENSION_STATES[id] || DIMENSION_STATES.papers;
  const state = stateGroup.states[stateKey] || stateGroup.states.default;
  return {
    module: stateGroup.label || id,
    panelStage: state.panelStage || '默认',
    before: state.before || '暂无输入维度说明',
    after: state.after || '暂无输出维度说明',
    detail: state.detail || '暂无补充说明',
    box: state.box || `${stateGroup.label || id}\n${state.panelStage || '默认'}\n查看右侧面板`,
  };
}

function refreshModuleDimensionSprites(activeId = null, activeStateKey = 'default') {
  moduleMap.forEach((module, id) => {
    if (!module.dimensionSprite || !module.dimensionSprite.userData.updateText) return;
    const state = getDimensionState(id, id === activeId ? activeStateKey : 'default');
    module.dimensionSprite.userData.updateText(state.box);
    module.dimensionSprite.material.opacity = 0;
  });
}

function setDimensionState(id, stateKey = 'default') {
  const state = getDimensionState(id, stateKey);
  dashboardState.focusId = id;
  refreshModuleDimensionSprites(moduleMap.has(id) ? id : null, stateKey);
  dashboardState.dimensionInfo = {
    module: state.module,
    stage: state.panelStage,
    before: state.before,
    after: state.after,
    detail: state.detail,
  };
  updateDashboard();
}

function createDiagramBox(spec) {
  const group = new THREE.Group();
  group.position.copy(spec.position);

  const depth = Math.min(spec.depth || 18, 10);
  const faceTexture = createModuleSurfaceTexture(spec, {
    width: Math.max(540, Math.round(spec.size.x * 3.4)),
    height: Math.max(200, Math.round(spec.size.y * 3.1)),
  });

  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(spec.size.x, spec.size.y, depth),
    [
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
      new THREE.MeshBasicMaterial({ map: faceTexture, transparent: true, depthWrite: false }),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
    ],
  );
  mesh.position.z = 0;
  mesh.userData.boxId = spec.id;
  mesh.renderOrder = 20;
  group.add(mesh);

  const edgeLines = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(spec.size.x, spec.size.y, depth)),
    new THREE.LineDashedMaterial({
      color: spec.border || '#f8fafc',
      transparent: true,
      opacity: 0.96,
      dashSize: 12,
      gapSize: 8,
    }),
  );
  edgeLines.computeLineDistances();
  edgeLines.position.z = 0.5;
  edgeLines.renderOrder = 21;
  group.add(edgeLines);

  const dimensionLabel = createTextSprite('等待高亮\n显示维度流转', {
    width: 720,
    height: 200,
    fontSize: 22,
    scale: 0.2,
    background: 'rgba(8,14,24,0.70)',
    border: 'rgba(255,255,255,0.10)',
  });
  dimensionLabel.position.set(0, -spec.size.y * 0.72, 10);
  dimensionLabel.material.opacity = 0;
  group.add(dimensionLabel);

  architectureGroup.add(group);
  mesh.userData.focusId = spec.id;
  moduleClickTargets.push(mesh);
  moduleMap.set(spec.id, {
    id: spec.id,
    label: spec.label,
    spec,
    group,
    mesh,
    edgeLines,
    dimensionSprite: dimensionLabel,
  });

  addModuleInteriorVisual(moduleMap.get(spec.id));

  focusTargets.set(spec.id, {
    position: spec.position.clone().add(spec.cameraOffset || new THREE.Vector3(0, 0, 420)),
    target: spec.position.clone(),
    label: spec.dashboardLabel || spec.label.replace(/\n/g, ''),
    kind: 'module',
  });
}

function createChartFrame({ position, size }) {
  const frame = new THREE.Mesh(
    new THREE.BoxGeometry(size.x, size.y, 6),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.02,
    }),
  );
  frame.position.copy(position);
  frame.position.z = -6;
  architectureGroup.add(frame);
  return frame;
}

function createArrowHead(position, rotationZ = 0, color = '#111111', scale = 1) {
  return null;
}

function createStaticPath(points, color = '#111111', opacity = 1) {
  const line = createLine(points, color, opacity);
  line.position.z = 5;
  line.visible = false;
  return line;
}

function createConnector(id, from, to, color = '#61dafb', lift = 0, bend = 0) {
  const start = typeof from === 'string'
    ? (from.endsWith('Paper') ? getPaperAnchor(from) : getModuleAnchor(from))
    : from.clone();
  const end = typeof to === 'string'
    ? (to.endsWith('Paper') ? getPaperAnchor(to) : getModuleAnchor(to))
    : to.clone();
  const mid = start.clone().lerp(end, 0.5);
  mid.y += lift;
  mid.x += bend;
  const curve = new THREE.CatmullRomCurve3([start, mid, end]);
  const points = curve.getPoints(70);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.26 }),
  );
  line.visible = false;
  connectorMap.set(id, { id, curve, line, material: line.material, color, from, to });
}

const chartSpecs = [
  {
    id: 'x_rp',
    label: 'X_rp',
    position: new THREE.Vector3(-700, -504, 0),
    size: new THREE.Vector3(136, 86),
    fill: '#d9ecff',
    haloColor: '#61dafb',
    fontSize: 28,
    dashboardLabel: 'X_rp 输入特征',
  },
  {
    id: 'x_spCtx',
    label: 'X_hist',
    position: new THREE.Vector3(-485, -504, 0),
    size: new THREE.Vector3(148, 86),
    fill: '#f8ebce',
    haloColor: '#fbbf24',
    fontSize: 26,
    dashboardLabel: 'X_hist 历史上下文',
  },
  {
    id: 'x_env',
    label: 'X_env',
    position: new THREE.Vector3(-245, -504, 0),
    size: new THREE.Vector3(136, 86),
    fill: '#dcfce7',
    haloColor: '#7ee787',
    fontSize: 28,
    dashboardLabel: 'X_env 环境特征',
  },
  {
    id: 'x_cand',
    label: 'X_cand',
    position: new THREE.Vector3(-25, -504, 0),
    size: new THREE.Vector3(148, 86),
    fill: '#ede9fe',
    haloColor: '#a78bfa',
    fontSize: 26,
    dashboardLabel: 'X_cand 候选模板',
  },
  {
    id: 'inputEmbedding',
    label: '输入嵌入层',
    position: new THREE.Vector3(-362, -238, 0),
    size: new THREE.Vector3(300, 164),
    fill: '#f9ecea',
    haloColor: '#f59e0b',
    fontSize: 48,
    renderMode: 'box',
    depth: 18,
    opacity: 0.44,
    dashboardLabel: '输入嵌入层',
  },
  {
    id: 'concatPanel',
    label: 'Concat',
    position: new THREE.Vector3(-362, -112, 0),
    size: new THREE.Vector3(230, 70),
    fill: '#f4f5f7',
    haloColor: '#94a3b8',
    fontSize: 26,
    renderMode: 'box',
    depth: 8,
    opacity: 0.38,
    dashboardLabel: 'Token Concat',
  },
  {
    id: 'encoder',
    label: '多头\n自注意力层',
    position: new THREE.Vector3(-362, 28, 0),
    size: new THREE.Vector3(250, 154),
    fill: '#f8ebce',
    haloColor: '#fbbf24',
    fontSize: 44,
    renderMode: 'box',
    depth: 18,
    opacity: 0.46,
    dashboardLabel: '编码器自注意力层',
  },
  {
    id: 'encoderAddNormBottom',
    label: 'Add & Norm',
    position: new THREE.Vector3(-362, 168, 0),
    size: new THREE.Vector3(250, 74),
    fill: '#fbf8c8',
    haloColor: '#fde68a',
    fontSize: 44,
    renderMode: 'box',
    depth: 12,
    opacity: 0.4,
    dashboardLabel: '编码器 Add & Norm',
  },
  {
    id: 'encoderFeedForward',
    label: '前馈\n网络层',
    position: new THREE.Vector3(-362, 284, 0),
    size: new THREE.Vector3(226, 56),
    fill: '#d9ecff',
    haloColor: '#93c5fd',
    fontSize: 34,
    renderMode: 'box',
    depth: 18,
    opacity: 0.44,
    dashboardLabel: '编码器前馈网络层',
  },
  {
    id: 'encoderAddNormTop',
    label: 'Add & Norm',
    position: new THREE.Vector3(-362, 398, 0),
    size: new THREE.Vector3(250, 74),
    fill: '#fbf8c8',
    haloColor: '#fde68a',
    fontSize: 44,
    renderMode: 'box',
    depth: 12,
    opacity: 0.4,
    dashboardLabel: '编码器 Add & Norm',
  },
  {
    id: 'outputEmbedding',
    label: 'Embedding&Concat',
    position: new THREE.Vector3(352, -322, 0),
    size: new THREE.Vector3(320, 138),
    fill: '#f9ecea',
    haloColor: '#f59e0b',
    fontSize: 42,
    renderMode: 'box',
    depth: 14,
    opacity: 0.42,
    dashboardLabel: 'Embedding&Concat',
  },
  {
    id: 'questionSetAttention',
    label: '题组集合\n自注意力层',
    position: new THREE.Vector3(352, -126, 0),
    size: new THREE.Vector3(280, 164),
    fill: '#f8ebce',
    haloColor: '#fbbf24',
    fontSize: 42,
    renderMode: 'box',
    depth: 18,
    opacity: 0.46,
    dashboardLabel: '题组集合自注意力层',
  },
  {
    id: 'decoderAddNormSelf',
    label: 'Add & Norm',
    position: new THREE.Vector3(352, 24, 0),
    size: new THREE.Vector3(250, 74),
    fill: '#fbf8c8',
    haloColor: '#fde68a',
    fontSize: 44,
    renderMode: 'box',
    depth: 12,
    opacity: 0.4,
    dashboardLabel: '解码器 Add & Norm',
  },
  {
    id: 'decoderCrossAttention',
    label: '多头\n交叉注意力层',
    position: new THREE.Vector3(352, 182, 0),
    size: new THREE.Vector3(280, 154),
    fill: '#f8ebce',
    haloColor: '#fbbf24',
    fontSize: 42,
    renderMode: 'box',
    depth: 18,
    opacity: 0.46,
    dashboardLabel: '多头交叉注意力层',
  },
  {
    id: 'decoderAddNormCross',
    label: 'Add & Norm',
    position: new THREE.Vector3(352, 332, 0),
    size: new THREE.Vector3(250, 74),
    fill: '#fbf8c8',
    haloColor: '#fde68a',
    fontSize: 44,
    renderMode: 'box',
    depth: 12,
    opacity: 0.4,
    dashboardLabel: '解码器 Add & Norm',
  },
  {
    id: 'decoder',
    label: '前馈\n网络层',
    position: new THREE.Vector3(352, 486, 0),
    size: new THREE.Vector3(250, 154),
    fill: '#d9ecff',
    haloColor: '#93c5fd',
    fontSize: 46,
    renderMode: 'box',
    depth: 18,
    opacity: 0.44,
    dashboardLabel: '解码器前馈网络层',
  },
  {
    id: 'decoderAddNormTop',
    label: 'Add & Norm',
    position: new THREE.Vector3(352, 636, 0),
    size: new THREE.Vector3(250, 74),
    fill: '#fbf8c8',
    haloColor: '#fde68a',
    fontSize: 44,
    renderMode: 'box',
    depth: 12,
    opacity: 0.4,
    dashboardLabel: '解码器 Add & Norm',
  },
  {
    id: 'maskHead',
    label: 'Mask\nHead',
    position: new THREE.Vector3(258, 836, 0),
    size: new THREE.Vector3(120, 56),
    fill: '#eee1c8',
    haloColor: '#fde68a',
    fontSize: 30,
    renderMode: 'box',
    depth: 12,
    opacity: 0.5,
    dashboardLabel: 'Mask Head',
  },
  {
    id: 'valueHead',
    label: 'Value\nHead',
    position: new THREE.Vector3(442, 836, 0),
    size: new THREE.Vector3(120, 56),
    fill: '#eee1c8',
    haloColor: '#fde68a',
    fontSize: 30,
    renderMode: 'box',
    depth: 12,
    opacity: 0.5,
    dashboardLabel: 'Value Head',
  },
  {
    id: 'scoreHead',
    label: 'Score\nHead',
    position: new THREE.Vector3(534, 762, 0),
    size: new THREE.Vector3(120, 56),
    fill: '#eee1c8',
    haloColor: '#fde68a',
    fontSize: 30,
    renderMode: 'box',
    depth: 12,
    opacity: 0.5,
    dashboardLabel: 'Score Head',
  },
  {
    id: 'countHead',
    label: 'Count\nHead',
    position: new THREE.Vector3(166, 762, 0),
    size: new THREE.Vector3(120, 56),
    fill: '#eee1c8',
    haloColor: '#fde68a',
    fontSize: 30,
    renderMode: 'box',
    depth: 12,
    opacity: 0.5,
    dashboardLabel: 'Count Head',
  },
  {
    id: 'slotSelectHead',
    label: 'Slot Select\nHead',
    position: new THREE.Vector3(350, 762, 0),
    size: new THREE.Vector3(144, 56),
    fill: '#eee1c8',
    haloColor: '#fde68a',
    fontSize: 26,
    renderMode: 'box',
    depth: 12,
    opacity: 0.5,
    dashboardLabel: 'Slot Select Head',
  },
  {
    id: 'outputBlock',
    label: '生成题组\nQuestion Block',
    position: new THREE.Vector3(760, 778, 0),
    size: new THREE.Vector3(210, 80),
    fill: '#f4f5f7',
    haloColor: '#7ee787',
    fontSize: 28,
    renderMode: 'box',
    depth: 14,
    opacity: 0.44,
    dashboardLabel: '题组输出',
    cameraOffset: new THREE.Vector3(90, 0, 360),
  },
  {
    id: 'weights',
    label: 'Weights',
    position: new THREE.Vector3(830, 204, 0),
    size: new THREE.Vector3(180, 88),
    fill: '#fff2cf',
    haloColor: '#fbbf24',
    fontSize: 40,
    renderMode: 'box',
    depth: 14,
    opacity: 0.44,
    dashboardLabel: '权重与检查点',
    cameraOffset: new THREE.Vector3(70, 0, 360),
  },
  {
    id: 'analysis',
    label: 'Analysis',
    position: new THREE.Vector3(830, 10, 0),
    size: new THREE.Vector3(180, 88),
    fill: '#dff2ff',
    haloColor: '#38bdf8',
    fontSize: 40,
    renderMode: 'box',
    depth: 14,
    opacity: 0.44,
    dashboardLabel: '反馈分析',
    cameraOffset: new THREE.Vector3(70, 0, 360),
  },
];

const DECODER_STACK_SHIFT_X = 102;
[
  'outputEmbedding',
  'questionSetAttention',
  'decoderAddNormSelf',
  'decoderCrossAttention',
  'decoderAddNormCross',
  'decoder',
  'decoderAddNormTop',
  'maskHead',
  'valueHead',
  'scoreHead',
  'countHead',
  'slotSelectHead',
  'outputBlock',
  'weights',
  'analysis',
].forEach((id) => {
  const spec = chartSpecs.find((item) => item.id === id);
  if (spec) spec.position.x += DECODER_STACK_SHIFT_X;
});

chartSpecs.forEach(createDiagramBox);
refreshModuleDimensionSprites();

const createStaticArrow = (points, color = '#111111', opacity = 1, scale = 0.88) => {
  if (!points || points.length < 2) return;
  createStaticPath(points, color, opacity);
  const from = points[points.length - 2];
  const to = points[points.length - 1];
  const rotationZ = Math.atan2(to.y - from.y, to.x - from.x) - Math.PI / 2;
  createArrowHead(new THREE.Vector3(to.x, to.y, 0), rotationZ, color, scale);
};

const moduleTopPoint = (id, gap = 10) => {
  const module = moduleMap.get(id);
  return new THREE.Vector3(module.group.position.x, module.group.position.y + module.spec.size.y / 2 + gap, 0);
};
const moduleBottomPoint = (id, gap = 10) => {
  const module = moduleMap.get(id);
  return new THREE.Vector3(module.group.position.x, module.group.position.y - module.spec.size.y / 2 - gap, 0);
};
const moduleLeftPoint = (id, ratio = 0.5, gap = 10) => {
  const module = moduleMap.get(id);
  return new THREE.Vector3(module.group.position.x - module.spec.size.x / 2 - gap, module.group.position.y + (ratio - 0.5) * module.spec.size.y, 0);
};
const moduleRightPoint = (id, ratio = 0.5, gap = 10) => {
  const module = moduleMap.get(id);
  return new THREE.Vector3(module.group.position.x + module.spec.size.x / 2 + gap, module.group.position.y + (ratio - 0.5) * module.spec.size.y, 0);
};
const moduleBottomPointAt = (id, ratio = 0.5, gap = 10) => {
  const module = moduleMap.get(id);
  return new THREE.Vector3(
    module.group.position.x + (ratio - 0.5) * module.spec.size.x,
    module.group.position.y - module.spec.size.y / 2 - gap,
    0,
  );
};
const moduleWorldLeftPointAt = (id, localY = 0, gap = 10, z = 10) => {
  const module = moduleMap.get(id);
  return getModuleWorldPoint(id, -module.spec.size.x / 2 - gap, localY, z);
};
const moduleWorldRightPointAt = (id, localY = 0, gap = 10, z = 10) => {
  const module = moduleMap.get(id);
  return getModuleWorldPoint(id, module.spec.size.x / 2 + gap, localY, z);
};
const moduleWorldBottomPointAt = (id, localX = 0, gap = 10, z = 10) => {
  const module = moduleMap.get(id);
  return getModuleWorldPoint(id, localX, -module.spec.size.y / 2 - gap, z);
};
const moduleWorldTopPointAt = (id, localX = 0, gap = 10, z = 10) => {
  const module = moduleMap.get(id);
  return getModuleWorldPoint(id, localX, module.spec.size.y / 2 + gap, z);
};
const createVerticalArrowBetween = (fromId, toId, startGap = 8, endGap = 8) => {
  const start = moduleTopPoint(fromId, startGap);
  const end = moduleBottomPoint(toId, endGap);
  createStaticArrow([
    start,
    new THREE.Vector3(end.x, end.y, 0),
  ], '#111111', 1, 0.9);
};

const encoderInputTargets = [0.18, 0.42, 0.62, 0.84].map((ratio) => {
  const module = moduleMap.get('inputEmbedding');
  return new THREE.Vector3(
    module.group.position.x + (ratio - 0.5) * module.spec.size.x,
    module.group.position.y - module.spec.size.y / 2 - 10,
    0,
  );
});
['x_rp', 'x_spCtx', 'x_env', 'x_cand'].forEach((id, index) => {
  const start = moduleTopPoint(id, 8);
  const target = encoderInputTargets[index];
  const elbowY = Math.max(start.y + 86, target.y + 64);
  createStaticArrow([
    start,
    new THREE.Vector3(start.x, elbowY, 0),
    new THREE.Vector3(target.x, elbowY, 0),
    target,
  ], '#111111', 1, 0.78);
});

createVerticalArrowBetween('inputEmbedding', 'concatPanel', 0, 0);
const concatOut = moduleTopPoint('concatPanel', 8);
const encoderForkTargets = [0.18, 0.5, 0.82].map((ratio) => {
  const module = moduleMap.get('encoder');
  return new THREE.Vector3(
    module.group.position.x + (ratio - 0.5) * module.spec.size.x,
    module.group.position.y - module.spec.size.y / 2 - 12,
    0,
  );
});
const forkHub = new THREE.Vector3(concatOut.x, concatOut.y + 18, 0);
createStaticPath([concatOut, forkHub], '#111111', 1);
createStaticArrow([forkHub, encoderForkTargets[1]], '#111111', 1, 0.82);
createStaticArrow([
  forkHub,
  new THREE.Vector3(encoderForkTargets[0].x, forkHub.y, 0),
  encoderForkTargets[0],
], '#111111', 1, 0.82);
createStaticArrow([
  forkHub,
  new THREE.Vector3(encoderForkTargets[2].x, forkHub.y, 0),
  encoderForkTargets[2],
], '#111111', 1, 0.82);
const concatResidualStart = moduleLeftPoint('concatPanel', 0.5, 8);
const concatResidualEnd = moduleLeftPoint('encoderAddNormBottom', 0.5, 10);
const concatResidualLaneX = moduleLeftPoint('encoder', 0.5, 12).x - 34;
createStaticArrow([
  concatResidualStart,
  new THREE.Vector3(concatResidualLaneX, concatResidualStart.y, 0),
  new THREE.Vector3(concatResidualLaneX, concatResidualEnd.y, 0),
  concatResidualEnd,
], '#111111', 1, 0.82);

const encoderToAddNormStart = moduleTopPoint('encoder', 0);
const encoderToAddNormEnd = moduleBottomPointAt('encoderAddNormBottom', 0.66, 0);
const encoderToAddNormElbowY = encoderToAddNormEnd.y - 18;
createStaticArrow([
  encoderToAddNormStart,
  new THREE.Vector3(encoderToAddNormStart.x, encoderToAddNormElbowY, 0),
  new THREE.Vector3(encoderToAddNormEnd.x, encoderToAddNormElbowY, 0),
  encoderToAddNormEnd,
], '#111111', 1, 0.9);
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('encoder', 52, 0, 18);
  const end = moduleWorldBottomPointAt('encoderAddNormBottom', 52, 0, 18);
  const elbowY = end.y - 18;
  return [
    start,
    new THREE.Vector3(start.x, elbowY, start.z),
    new THREE.Vector3(end.x, elbowY, start.z),
    end,
  ];
}, {
  color: '#7dd3fc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldLeftPointAt('concatPanel', 0, 8, 18);
  const end = moduleWorldLeftPointAt('encoderAddNormBottom', -6, 6, 18);
  const laneX = moduleWorldLeftPointAt('encoder', 0, 52, 18).x;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#93c5fd',
  opacity: 0.86,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('encoderAddNormBottom', 0, 0, 18);
  const end = moduleWorldBottomPointAt('encoderFeedForward', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('encoderFeedForward', 0, 0, 18);
  const end = moduleWorldBottomPointAt('encoderAddNormTop', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldLeftPointAt('encoderAddNormBottom', 10, 0, 18);
  const end = moduleWorldLeftPointAt('encoderAddNormTop', 0, 0, 18);
  const laneX = moduleWorldLeftPointAt('encoder', 0, 34, 18).x;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#fde68a',
  opacity: 0.86,
  shaftWidth: 4.8,
  headWidth: 11,
  headLength: 14,
  depth: 3.2,
});
createVerticalArrowBetween('encoderAddNormBottom', 'encoderFeedForward', 10, 16);
createVerticalArrowBetween('encoderFeedForward', 'encoderAddNormTop', 10, 16);

createLiveArrow(() => {
  const start = moduleWorldTopPointAt('outputEmbedding', 0, 0, 18);
  const end = moduleWorldBottomPointAt('questionSetAttention', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldLeftPointAt('outputEmbedding', 0, 8, 18);
  const end = moduleWorldLeftPointAt('decoderAddNormSelf', -4, 8, 18);
  const laneX = Math.min(start.x, end.x) - 34;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#93c5fd',
  opacity: 0.86,
  shaftWidth: 5,
  headWidth: 11,
  headLength: 14,
  depth: 3.2,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('questionSetAttention', 0, 0, 18);
  const end = moduleWorldBottomPointAt('decoderAddNormSelf', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('decoderAddNormSelf', 0, 0, 18);
  const end = moduleWorldBottomPointAt('decoderCrossAttention', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldRightPointAt('encoderAddNormTop', 6, 8, 18);
  const end = moduleWorldLeftPointAt('decoderCrossAttention', 0, 8, 18);
  const laneX = 18;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#7dd3fc',
  opacity: 0.86,
  shaftWidth: 5,
  headWidth: 11,
  headLength: 14,
  depth: 3.2,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('decoderCrossAttention', 0, 0, 18);
  const end = moduleWorldBottomPointAt('decoderAddNormCross', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldLeftPointAt('decoderAddNormSelf', 2, 8, 18);
  const end = moduleWorldLeftPointAt('decoderAddNormCross', -2, 8, 18);
  const laneX = Math.min(start.x, end.x) - 32;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#fde68a',
  opacity: 0.86,
  shaftWidth: 4.8,
  headWidth: 11,
  headLength: 14,
  depth: 3.2,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('decoderAddNormCross', 0, 0, 18);
  const end = moduleWorldBottomPointAt('decoder', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldTopPointAt('decoder', 0, 0, 18);
  const end = moduleWorldBottomPointAt('decoderAddNormTop', 0, 0, 18);
  return [start, end];
}, {
  color: '#f8fafc',
  opacity: 0.88,
  shaftWidth: 5.2,
  headWidth: 12,
  headLength: 15,
  depth: 3.4,
});
createLiveArrow(() => {
  const start = moduleWorldLeftPointAt('decoderAddNormCross', 2, 8, 18);
  const end = moduleWorldLeftPointAt('decoderAddNormTop', -2, 8, 18);
  const laneX = Math.min(start.x, end.x) - 32;
  return [
    start,
    new THREE.Vector3(laneX, start.y, start.z),
    new THREE.Vector3(laneX, end.y, start.z),
    end,
  ];
}, {
  color: '#fde68a',
  opacity: 0.86,
  shaftWidth: 4.8,
  headWidth: 11,
  headLength: 14,
  depth: 3.2,
});
[
  ['countHead', -112],
  ['slotSelectHead', -56],
  ['scoreHead', 0],
  ['maskHead', 56],
  ['valueHead', 112],
].forEach(([headId, localX]) => {
  createLiveArrow(() => {
    const start = moduleWorldTopPointAt('decoderAddNormTop', localX, 0, 18);
    const end = moduleWorldBottomPointAt(headId, 0, 0, 18);
    const hubY = start.y + 28;
    return [
      start,
      new THREE.Vector3(start.x, hubY, start.z),
      new THREE.Vector3(end.x, hubY, start.z),
      end,
    ];
  }, {
    color: '#f8fafc',
    opacity: 0.86,
    shaftWidth: 4.8,
    headWidth: 11,
    headLength: 14,
    depth: 3.2,
  });
});

const outputBlockLeftUpper = moduleLeftPoint('outputBlock', 0.34, 10);
const outputBlockLeftLower = moduleLeftPoint('outputBlock', 0.68, 10);
const outputBlockLeftMiddle = moduleLeftPoint('outputBlock', 0.52, 10);
createStaticArrow([
  moduleRightPoint('valueHead', 0.5, 8),
  new THREE.Vector3(outputBlockLeftUpper.x - 46, moduleRightPoint('valueHead', 0.5, 8).y, 0),
  new THREE.Vector3(outputBlockLeftUpper.x - 46, outputBlockLeftUpper.y, 0),
  outputBlockLeftUpper,
], '#111111', 1, 0.78);
createStaticArrow([
  moduleRightPoint('slotSelectHead', 0.5, 8),
  new THREE.Vector3(outputBlockLeftMiddle.x - 32, moduleRightPoint('slotSelectHead', 0.5, 8).y, 0),
  new THREE.Vector3(outputBlockLeftMiddle.x - 32, outputBlockLeftMiddle.y, 0),
  outputBlockLeftMiddle,
], '#111111', 1, 0.78);
createStaticArrow([
  moduleRightPoint('countHead', 0.5, 8),
  new THREE.Vector3(outputBlockLeftLower.x - 18, moduleRightPoint('countHead', 0.5, 8).y, 0),
  new THREE.Vector3(outputBlockLeftLower.x - 18, outputBlockLeftLower.y, 0),
  outputBlockLeftLower,
], '#111111', 1, 0.78);

const encoderRouteStart = moduleRightPoint('encoderAddNormTop', 0.2, 10);
const encoderRouteElbow = new THREE.Vector3(70, encoderRouteStart.y, 0);
const encoderRouteDrop = new THREE.Vector3(70, moduleLeftPoint('decoderCrossAttention', 0.5, 10).y, 0);
const encoderRouteEnd = moduleLeftPoint('decoderCrossAttention', 0.5, 10);

function drawSurveyCard(ctx, x, y, width, height, label, value, {
  labelColor = '#b8c2d8',
  valueColor = '#f8fbff',
  fill = 'rgba(15,23,42,0.82)',
  border = 'rgba(148,163,184,0.55)',
} = {}) {
  ctx.fillStyle = fill;
  ctx.strokeStyle = border;
  ctx.lineWidth = 2;
  drawRoundRect(ctx, x, y, width, height, 18);
  ctx.fill();
  ctx.stroke();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = labelColor;
  ctx.font = '600 24px "PingFang SC", Arial, sans-serif';
  ctx.fillText(label, x + 22, y + 16);

  ctx.fillStyle = valueColor;
  ctx.font = '700 28px "PingFang SC", Arial, sans-serif';
  const lines = String(value || '-').split('\n');
  lines.slice(0, 4).forEach((line, index) => {
    ctx.fillText(line, x + 22, y + 50 + index * 34);
  });
}

function createSurveyBoardTexture(paper, {
  width = 1120,
  height = 1480,
  ghost = false,
  showBorder = true,
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const accent = paper.kind === 'trip' ? '#61dafb' : '#a78bfa';
  const accentSoft = paper.kind === 'trip' ? 'rgba(97,218,251,0.18)' : 'rgba(167,139,250,0.18)';
  const bodyFill = ghost ? 'rgba(24,39,63,0.42)' : 'rgba(9,16,28,0.90)';
  const border = ghost ? 'rgba(148,163,184,0.48)' : 'rgba(226,232,240,0.82)';

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = ghost ? 'rgba(0,0,0,0.05)' : 'rgba(8,15,28,0.24)';
  drawRoundRect(ctx, 38, 42, width - 76, height - 84, 42);
  ctx.fill();

  ctx.fillStyle = bodyFill;
  ctx.strokeStyle = border;
  ctx.lineWidth = ghost ? 4 : 6;
  drawRoundRect(ctx, 18, 18, width - 36, height - 36, 38);
  ctx.fill();
  if (showBorder) ctx.stroke();

  const gradient = ctx.createLinearGradient(56, 0, width - 56, height);
  gradient.addColorStop(0, paper.kind === 'trip' ? 'rgba(30,64,175,0.18)' : 'rgba(109,40,217,0.18)');
  gradient.addColorStop(1, 'rgba(15,23,42,0.02)');
  ctx.fillStyle = gradient;
  drawRoundRect(ctx, 42, 42, width - 84, height - 84, 34);
  ctx.fill();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  ctx.fillStyle = '#f8fbff';
  ctx.font = '700 50px "PingFang SC", Arial, sans-serif';
  ctx.fillText(paper.kind === 'trip' ? 'Trip Diary Input Board' : 'SP Survey Decision Board', 56, 170);

  ctx.fillStyle = '#b8c2d8';
  ctx.font = '500 26px "PingFang SC", Arial, sans-serif';
  ctx.fillText(
    paper.kind === 'trip'
      ? 'Respondent / trip baseline information'
      : 'Generated scenario block and respondent feedback',
    56,
    220,
  );

  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(56, 254);
  ctx.lineTo(width - 56, 254);
  ctx.stroke();

  if (paper.kind === 'trip') {
    drawSurveyCard(ctx, 56, 290, 480, 120, '调查日期', paper.rows.date.value || '未填写');
    drawSurveyCard(ctx, 584, 290, 480, 120, '成员编号', paper.rows.member.value || '未填写');
    drawSurveyCard(ctx, 56, 438, 1008, 120, '出发地', paper.rows.origin.value || '未填写');
    drawSurveyCard(ctx, 56, 586, 1008, 120, '目的地', paper.rows.destination.value || '未填写');
    drawSurveyCard(ctx, 56, 734, 480, 120, '出发时间', paper.rows.depart.value || '未填写');
    drawSurveyCard(ctx, 584, 734, 480, 120, '到达时间', paper.rows.arrive.value || '未填写');

    const mode = paper.rows.modeMetro.checked ? '地铁'
      : paper.rows.modeBus.checked ? '公交'
        : paper.rows.modeCar.checked ? '小汽车'
          : paper.rows.modeWalk.checked ? '步行 / 骑行'
            : '未选择';
    const flags = [
      `换乘: ${paper.rows.transfer.checked ? '是' : '否'}`,
      `接送: ${paper.rows.escort.checked ? '是' : '否'}`,
    ].join('    ');
    drawSurveyCard(ctx, 56, 900, 1008, 136, '方式与链条', `${mode}\n${flags}`, {
      fill: 'rgba(12,31,52,0.86)',
      border: 'rgba(97,218,251,0.58)',
    });
    ctx.fillStyle = '#9fb0c8';
    ctx.font = '500 24px "PingFang SC", Arial, sans-serif';
    ctx.fillText('Encoded into X_rp and X_env before entering the encoder.', 56, 1118);
  } else {
    const reasons = [
      paper.rows.reasonTime.checked ? '时间更稳定' : null,
      paper.rows.reasonComfort.checked ? '舒适性更高' : null,
      paper.rows.reasonFare.checked ? '费用可接受' : null,
      paper.rows.reasonTransfer.checked ? '换乘可接受' : null,
    ].filter(Boolean);
    const selectedPlan = paper.rows.planB.checked ? '方案 B'
      : paper.rows.planA.checked ? '方案 A'
        : '未选择';

    drawSurveyCard(ctx, 56, 290, 480, 120, '受访者编号', paper.rows.spId.value || '未填写');
    drawSurveyCard(ctx, 584, 290, 480, 120, '当前场景', paper.refs.scenarioTag.textContent || 'SP part · G0');
    drawSurveyCard(ctx, 56, 438, 480, 196, '方案 A', paper.refs.planAItems.map((el) => `• ${el.textContent}`).join('\n'));
    drawSurveyCard(ctx, 584, 438, 480, 196, '方案 B', paper.refs.planBItems.map((el) => `• ${el.textContent}`).join('\n'));
    drawSurveyCard(ctx, 56, 662, 1008, 120, '当前选择', selectedPlan, {
      fill: 'rgba(36,18,66,0.88)',
      border: 'rgba(167,139,250,0.62)',
    });
    drawSurveyCard(ctx, 56, 810, 1008, 144, '原因勾选', reasons.length ? reasons.join('\n') : '尚未勾选');
    drawSurveyCard(ctx, 56, 982, 1008, 180, '补充说明', paper.rows.note.value || '未填写');
    ctx.fillStyle = '#9fb0c8';
    ctx.font = '500 24px "PingFang SC", Arial, sans-serif';
    ctx.fillText('Used for analysis, parameter refresh and the next generated block.', 56, 1248);
  }

  if (ghost) {
    ctx.fillStyle = 'rgba(226,232,240,0.10)';
    for (let i = 0; i < 7; i += 1) {
      const lineW = width - 160 - (i % 3) * 120;
      drawRoundRect(ctx, 56, 330 + i * 122, lineW, 22, 11);
      ctx.fill();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createPaperBoardMesh(paper, {
  title,
  ghost = false,
  showFaceBorder = true,
  showEdgeLines = true,
} = {}) {
  const group = new THREE.Group();
  const width = 700;
  const height = 920;
  const depth = ghost ? 20 : 34;
  const faceTexture = createSurveyBoardTexture(paper, { ghost, showBorder: showFaceBorder });

  const highlight = new THREE.Mesh(
    new THREE.BoxGeometry(width + 40, height + 40, depth + 14),
    new THREE.MeshBasicMaterial({
      color: paper.kind === 'trip' ? 0x60a5fa : 0xa78bfa,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  highlight.position.z = -4;
  if (!ghost) group.add(highlight);

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    [
      new THREE.MeshPhongMaterial({
        color: paper.kind === 'trip' ? (ghost ? 0x1e3a5f : 0x10243a) : (ghost ? 0x3b2468 : 0x22153f),
        transparent: false,
        opacity: 1,
        shininess: 65,
        emissive: new THREE.Color(paper.kind === 'trip' ? '#0b1626' : '#140d24'),
        emissiveIntensity: ghost ? 0.10 : 0.18,
      }),
      new THREE.MeshPhongMaterial({
        color: paper.kind === 'trip' ? (ghost ? 0x1e3a5f : 0x10243a) : (ghost ? 0x3b2468 : 0x22153f),
        transparent: false,
        opacity: 1,
        shininess: 65,
        emissive: new THREE.Color(paper.kind === 'trip' ? '#0b1626' : '#140d24'),
        emissiveIntensity: ghost ? 0.10 : 0.18,
      }),
      new THREE.MeshPhongMaterial({
        color: paper.kind === 'trip' ? (ghost ? 0x1e3a5f : 0x10243a) : (ghost ? 0x3b2468 : 0x22153f),
        transparent: false,
        opacity: 1,
        shininess: 65,
        emissive: new THREE.Color(paper.kind === 'trip' ? '#0b1626' : '#140d24'),
        emissiveIntensity: ghost ? 0.10 : 0.18,
      }),
      new THREE.MeshPhongMaterial({
        color: paper.kind === 'trip' ? (ghost ? 0x1e3a5f : 0x10243a) : (ghost ? 0x3b2468 : 0x22153f),
        transparent: false,
        opacity: 1,
        shininess: 65,
        emissive: new THREE.Color(paper.kind === 'trip' ? '#0b1626' : '#140d24'),
        emissiveIntensity: ghost ? 0.10 : 0.18,
      }),
      new THREE.MeshBasicMaterial({ map: faceTexture, transparent: false }),
      new THREE.MeshBasicMaterial({ map: faceTexture.clone(), transparent: false }),
    ],
  );
  body.renderOrder = ghost ? 4 : 6;
  group.add(body);

  let edgeLines = null;
  if (showEdgeLines) {
    edgeLines = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth)),
      new THREE.LineBasicMaterial({
        color: ghost ? '#94a3b8' : '#0f172a',
        transparent: false,
        opacity: 1,
      }),
    );
    edgeLines.position.z = 0.5;
    edgeLines.renderOrder = ghost ? 5 : 7;
    group.add(edgeLines);
  }

  // Do not add floating RP/SP title tags above the paper; keep arrow area clean.

  return { group, body, highlight, width, height, depth };
}

function createTripDiaryPaper() {
  const root = document.createElement('div');
  root.className = 'paper';
  root.innerHTML = `
    <div class="pen-dot"></div>
    <header class="paper-header">
      <div>
        <span class="paper-tag">RP part · Trip Diary</span>
        <h2 class="paper-title">出行日志记录</h2>
        <p class="paper-subtitle">模拟一次出行记录的填写过程，包括 RP 基础信息、时间、地点与方式选择。</p>
      </div>
    </header>
    <div class="paper-form">
      <div class="section-title">RP / Basic Info</div>
      <div class="field-row" data-field="date-row">
        <label for="trip-date">调查日期</label>
        <input id="trip-date" type="text" value="" />
      </div>
      <div class="field-row" data-field="member-row">
        <label for="trip-member">成员编号</label>
        <input id="trip-member" type="text" value="" />
      </div>
      <div class="field-row" data-field="origin-row">
        <label for="trip-origin">出发地</label>
        <input id="trip-origin" type="text" value="" />
      </div>
      <div class="field-row" data-field="destination-row">
        <label for="trip-destination">目的地</label>
        <input id="trip-destination" type="text" value="" />
      </div>
      <div class="field-row" data-field="depart-row">
        <label for="trip-depart">出发时间</label>
        <input id="trip-depart" type="text" value="" />
      </div>
      <div class="field-row" data-field="arrive-row">
        <label for="trip-arrive">到达时间</label>
        <input id="trip-arrive" type="text" value="" />
      </div>

      <div class="section-title">Mode & Trip Chain</div>
      <div class="radio-grid" data-field="mode-group">
        <label data-choice="bus-card"><input type="radio" name="trip-mode" id="trip-mode-bus" />公交</label>
        <label data-choice="metro-card"><input type="radio" name="trip-mode" id="trip-mode-metro" />地铁</label>
        <label data-choice="car-card"><input type="radio" name="trip-mode" id="trip-mode-car" />小汽车</label>
        <label data-choice="walk-card"><input type="radio" name="trip-mode" id="trip-mode-walk" />步行/骑行</label>
      </div>
      <div class="check-grid" data-field="flags-group">
        <label data-choice="transfer-card"><input type="checkbox" id="trip-transfer" />本次出行包含换乘</label>
        <label data-choice="escort-card"><input type="checkbox" id="trip-escort" />本次为接送家人</label>
      </div>
    </div>
  `;

  return {
    kind: 'trip',
    root,
    pen: root.querySelector('.pen-dot'),
    rows: {
      date: root.querySelector('#trip-date'),
      member: root.querySelector('#trip-member'),
      origin: root.querySelector('#trip-origin'),
      destination: root.querySelector('#trip-destination'),
      depart: root.querySelector('#trip-depart'),
      arrive: root.querySelector('#trip-arrive'),
      modeBus: root.querySelector('#trip-mode-bus'),
      modeMetro: root.querySelector('#trip-mode-metro'),
      modeCar: root.querySelector('#trip-mode-car'),
      modeWalk: root.querySelector('#trip-mode-walk'),
      transfer: root.querySelector('#trip-transfer'),
      escort: root.querySelector('#trip-escort'),
    },
  };
}

function createSpSurveyPaper() {
  const root = document.createElement('div');
  root.className = 'paper';
  root.innerHTML = `
    <div class="pen-dot"></div>
    <header class="paper-header">
      <div>
        <span class="paper-tag" id="spScenarioTag">SP part · G0</span>
        <h2 class="paper-title">情景选择问卷</h2>
        <p class="paper-subtitle" id="spScenarioSubtitle">模拟一页 SP 题面，包括方案单选、原因勾选与补充说明。</p>
      </div>
    </header>
    <div class="paper-form">
      <div class="field-row" data-field="sp-id-row">
        <label for="sp-id">受访者编号</label>
        <input id="sp-id" type="text" value="" />
      </div>

      <div class="section-title">Question Block 01</div>
      <div class="choice-grid">
        <label class="option-card" data-choice="plan-a-card">
          <header>
            <h3>方案 A</h3>
            <input type="radio" name="sp-plan" id="plan-a" />
          </header>
          <ul class="plan-a-list">
            <li>公交接驳 5 分钟</li>
            <li>车内时间 14 分钟</li>
            <li>费用 3 元</li>
          </ul>
        </label>
        <label class="option-card" data-choice="plan-b-card">
          <header>
            <h3>方案 B</h3>
            <input type="radio" name="sp-plan" id="plan-b" />
          </header>
          <ul class="plan-b-list">
            <li>地铁接驳 7 分钟</li>
            <li>车内时间 11 分钟</li>
            <li>费用 4 元</li>
          </ul>
        </label>
      </div>

      <div class="section-title">Reasoning</div>
      <div class="check-grid" data-field="reason-group">
        <label data-choice="reason-time-card"><input type="checkbox" id="reason-time" />时间更稳定</label>
        <label data-choice="reason-comfort-card"><input type="checkbox" id="reason-comfort" />舒适性更高</label>
        <label data-choice="reason-fare-card"><input type="checkbox" id="reason-fare" />费用可接受</label>
        <label data-choice="reason-transfer-card"><input type="checkbox" id="reason-transfer" />换乘可接受</label>
      </div>

      <div class="field-row" data-field="sp-note-row">
        <label for="sp-note">补充说明</label>
        <textarea id="sp-note"></textarea>
      </div>
    </div>
  `;

  return {
    kind: 'sp',
    root,
    pen: root.querySelector('.pen-dot'),
    rows: {
      spId: root.querySelector('#sp-id'),
      planA: root.querySelector('#plan-a'),
      planB: root.querySelector('#plan-b'),
      reasonTime: root.querySelector('#reason-time'),
      reasonComfort: root.querySelector('#reason-comfort'),
      reasonFare: root.querySelector('#reason-fare'),
      reasonTransfer: root.querySelector('#reason-transfer'),
      note: root.querySelector('#sp-note'),
    },
    refs: {
      scenarioTag: root.querySelector('#spScenarioTag'),
      scenarioSubtitle: root.querySelector('#spScenarioSubtitle'),
      planAItems: Array.from(root.querySelectorAll('.plan-a-list li')),
      planBItems: Array.from(root.querySelectorAll('.plan-b-list li')),
    },
  };
}

function createPaperStack({ createMain, title, position, rotation, ghostSign = 1, scale = 1 }) {
  const group = new THREE.Group();
  group.position.copy(position);
  group.rotation.copy(rotation);
  group.scale.setScalar(scale);
  group.visible = false;
  webglScene.add(group);

  const paper = createMain();

  // Keep all back sheets arranged toward the left to avoid overlap blocking.
  const ghostOffsets = [
    new THREE.Vector3(-82, 28, -58),
    new THREE.Vector3(-42, 14, -30),
  ];
  const ghosts = ghostOffsets.map((offset, idx) => {
    const board = createPaperBoardMesh(paper, { title, ghost: true });
    board.group.position.copy(offset);
    board.group.rotation.z = -Math.abs(ghostSign) * (0.028 + idx * 0.018);
    board.group.scale.setScalar(0.97 - idx * 0.028);
    group.add(board.group);
    return board.group;
  });

  const board = createPaperBoardMesh(paper, { title });
  group.add(board.group);

  const refreshVisual = () => {
    const frontMaterial = Array.isArray(board.body.material) ? board.body.material[4] : null;
    const backMaterial = Array.isArray(board.body.material) ? board.body.material[5] : null;
    const nextMap = createSurveyBoardTexture(paper);
    if (frontMaterial?.map) frontMaterial.map.dispose();
    if (backMaterial?.map) backMaterial.map.dispose();
    if (frontMaterial) {
      frontMaterial.map = nextMap;
      frontMaterial.needsUpdate = true;
    }
    if (backMaterial) {
      backMaterial.map = nextMap.clone();
      backMaterial.needsUpdate = true;
    }
  };
  refreshVisual();

  return {
    ...paper,
    group,
    boardGroup: board.group,
    boardBody: board.body,
    boardHighlight: board.highlight,
    ghosts,
    basePosition: position.clone(),
    baseRotation: rotation.clone(),
    boardWidth: board.width,
    boardHeight: board.height,
    boardDepth: board.depth,
    refreshVisual,
    scale,
  };
}

function createThinInfoPanel({
  text,
  width = 150,
  height = 52,
  depth = 8,
  fill = '#f8f4e8',
  border = '#111827',
  textColor = '#111827',
  fontSize = 28,
  opacity = 0.94,
  shininess = 30,
} = {}) {
  const group = new THREE.Group();
  const texture = createPanelTexture(text, {
    width: Math.max(520, Math.round(width * 3.3)),
    height: Math.max(180, Math.round(height * 3.3)),
    fill,
    border: 'rgba(0,0,0,0)',
    textColor,
    fontSize,
    borderWidth: 0,
    radius: 20,
    shadow: false,
  });
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const sideMaterial = new THREE.MeshPhongMaterial({
    color: fill,
    transparent: true,
    opacity,
    shininess,
  });
  const box = new THREE.Mesh(
    geometry,
    [
      sideMaterial.clone(),
      sideMaterial.clone(),
      sideMaterial.clone(),
      sideMaterial.clone(),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity }),
      new THREE.MeshBasicMaterial({ map: texture.clone(), transparent: true, opacity }),
    ],
  );
  box.renderOrder = 26;
  group.add(box);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: border,
      transparent: true,
      opacity: 0.92,
    }),
  );
  edges.position.z = 0.6;
  edges.renderOrder = 27;
  group.add(edges);

  return {
    group,
    box,
    edges,
    width,
    height,
    depth,
  };
}

function createMiniVariableTokenPanel({
  title = 'X_rp',
  shapeFormula = 'X_{rp} \\in \\mathbb{R}^{B\\times d_{rp}}',
  inputFormula = null,
  outputFormula = null,
  width = 138,
  height = 74,
  depth = 8,
  fill = '#d9ecff',
  border = '#cbd5e1',
  cubeColor = fill,
  tokenLabels = ['v_1', 'v_2', 'v_3', 'v_4'],
  historical = false,
} = {}) {
  const group = new THREE.Group();

  const hitBox = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.02,
      depthWrite: false,
    }),
  );
  hitBox.renderOrder = 24;
  group.add(hitBox);

  const dashedEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth)),
    new THREE.LineDashedMaterial({
      color: border,
      transparent: true,
      opacity: 0.82,
      dashSize: 7,
      gapSize: 5,
    }),
  );
  dashedEdges.computeLineDistances();
  dashedEdges.position.z = 0.8;
  dashedEdges.renderOrder = 25;
  group.add(dashedEdges);

  const titleTag = createOverlaySprite(`[${title}]`, {
    width: Math.max(82, Math.round(width * 0.42)),
    height: 22,
    fontSize: 11,
    scale: 0.078,
    fill: '#f8fafc',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  titleTag.position.set(-width * 0.28, height * 0.35, 7);
  group.add(titleTag);

  const topFormulaTag = createFormulaSprite(outputFormula || shapeFormula, {
    width: Math.max(122, Math.round(width * 0.82)),
    height: 26,
    fontSize: 9.5,
    scale: 0.068,
    align: 'right',
    fill: '#f8fafc',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  topFormulaTag.position.set(width * 0.08, height * 0.33, 7);
  group.add(topFormulaTag);

  const bottomFormulaTag = createFormulaSprite(inputFormula || shapeFormula, {
    width: Math.max(122, Math.round(width * 0.82)),
    height: 26,
    fontSize: 9.5,
    scale: 0.068,
    align: 'left',
    fill: '#f8fafc',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  bottomFormulaTag.position.set(-width * 0.08, -height * 0.33, 7);
  group.add(bottomFormulaTag);

  const interior = new THREE.Group();
  const cubes = [];
  const baseTokenCount = Math.min(4, tokenLabels.length || 4);
  const stepX = width * 0.16;
  const startX = -stepX * ((baseTokenCount - 1) * 0.5);
  const tokenY = height * 0.02;
  const tokenZ = 8;
  const cubeSize = Math.min(14, height * 0.22);

  for (let i = 0; i < baseTokenCount; i += 1) {
    const x = startX + stepX * i;
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize),
      new THREE.MeshPhongMaterial({
        color: cubeColor,
        transparent: true,
        opacity: 0.58,
        emissive: new THREE.Color(cubeColor),
        emissiveIntensity: 0.18,
        shininess: 72,
      }),
    );
    cube.position.set(x, tokenY, tokenZ);
    interior.add(cube);
    cubes.push(cube);

    if (historical) {
      [1, 2, 3].forEach((ghostIdx) => {
        const ghost = new THREE.Mesh(
          new THREE.BoxGeometry(cubeSize * 0.92, cubeSize * 0.92, cubeSize * 0.92),
          new THREE.MeshPhongMaterial({
            color: cubeColor,
            transparent: true,
            opacity: ghostIdx === 1 ? 0.18 : (ghostIdx === 2 ? 0.11 : 0.07),
            emissive: new THREE.Color(cubeColor),
            emissiveIntensity: 0.06,
            shininess: 32,
            depthWrite: false,
          }),
        );
        ghost.position.set(x + ghostIdx * 4, tokenY + ghostIdx * 1.5, tokenZ - ghostIdx * 10);
        interior.add(ghost);
      });
    }

    const label = createOverlaySprite(tokenLabels[i] || `v_${i + 1}`, {
      width: 46,
      height: 16,
      fontSize: 8,
      scale: 0.07,
      fill: '#cbd5e1',
      background: 'rgba(15,23,42,0.38)',
      border: 'rgba(148,163,184,0.12)',
    });
    label.position.set(x, -height * 0.16, 7);
    interior.add(label);
  }

  group.add(interior);

  return {
    group,
    box: hitBox,
    edges: dashedEdges,
    width,
    height,
    depth,
    animate(elapsed = 0) {
      cubes.forEach((cube, index) => {
        const s = 1 + Math.sin(elapsed * 1.8 + index * 0.55) * 0.05;
        cube.scale.setScalar(s);
      });
    },
  };
}

function getPanelWorldPoint(panel, localX = 0, localY = 0, localZ = 0) {
  if (!panel?.group) return new THREE.Vector3();
  return panel.group.localToWorld(new THREE.Vector3(localX, localY, localZ));
}

function createDynamicTextBoard({
  title = 'Question Block',
  lines = [],
  width = 320,
  height = 220,
  depth = 8,
  fill = '#f8fafc',
  border = '#111827',
  titleColor = '#111827',
  textColor = '#111827',
  accentColor = '#0f172a',
  opacity = 0.96,
  titleFontSize = 34,
  textFontSize = 23,
  maxLinesPerColumn = 6,
  numbered = true,
  titleFontFamily = '"Avenir Next", "PingFang SC", Arial, sans-serif',
  textFontFamily = '"PingFang SC", Arial, sans-serif',
  resolutionScale = 4.2,
} = {}) {
  const group = new THREE.Group();
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(960, Math.round(width * resolutionScale));
  canvas.height = Math.max(720, Math.round(height * resolutionScale));
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const geometry = new THREE.BoxGeometry(width, height, depth);
  const sideMaterial = new THREE.MeshPhongMaterial({
    color: fill,
    transparent: true,
    opacity,
    shininess: 36,
  });
  const faceMaterial = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity,
  });
  const box = new THREE.Mesh(
    geometry,
    [
      sideMaterial.clone(),
      sideMaterial.clone(),
      sideMaterial.clone(),
      sideMaterial.clone(),
      faceMaterial,
      new THREE.MeshBasicMaterial({ map: texture.clone(), transparent: true, opacity }),
    ],
  );
  box.renderOrder = 26;
  group.add(box);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: border,
      transparent: true,
      opacity: 0.92,
    }),
  );
  edges.position.z = 0.6;
  edges.renderOrder = 27;
  group.add(edges);

  const draw = ({
    title: nextTitle = title,
    lines: nextLines = lines,
    footer = '',
  } = {}) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = fill;
    ctx.strokeStyle = border;
    ctx.lineWidth = 10;
    drawRoundRect(ctx, 10, 10, canvas.width - 20, canvas.height - 20, 38);
    ctx.fill();
    ctx.stroke();

    const padX = 48;
    const padTop = 40;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = titleColor;
    ctx.font = `700 ${titleFontSize}px ${titleFontFamily}`;
    ctx.fillText(nextTitle, padX, padTop);

    ctx.strokeStyle = 'rgba(15,23,42,0.18)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(padX, padTop + titleFontSize + 18);
    ctx.lineTo(canvas.width - padX, padTop + titleFontSize + 18);
    ctx.stroke();

    const visibleLines = Array.isArray(nextLines) ? nextLines.filter(Boolean) : [];
    const splitIntoTwoCols = visibleLines.length > maxLinesPerColumn;
    const firstColumn = splitIntoTwoCols ? visibleLines.slice(0, Math.ceil(visibleLines.length / 2)) : visibleLines;
    const secondColumn = splitIntoTwoCols ? visibleLines.slice(Math.ceil(visibleLines.length / 2)) : [];
    const columnWidth = splitIntoTwoCols
      ? (canvas.width - padX * 2 - 28) / 2
      : canvas.width - padX * 2;
    const colGap = splitIntoTwoCols ? 28 : 0;
    const lineHeight = Math.max(28, textFontSize * 1.25);
    const startY = padTop + titleFontSize + 42;

    const drawColumn = (columnLines, colIndex) => {
      const baseX = padX + colIndex * (columnWidth + colGap);
      columnLines.forEach((line, idx) => {
        const y = startY + idx * lineHeight;
        ctx.fillStyle = idx === columnLines.length - 1 ? accentColor : textColor;
        ctx.font = `600 ${textFontSize}px ${textFontFamily}`;
        if (numbered) {
          const prefix = `${colIndex === 0 ? idx + 1 : idx + 1 + firstColumn.length}.`;
          ctx.fillText(prefix, baseX, y);
          ctx.fillText(line, baseX + 38, y);
        } else {
          ctx.fillText(line, baseX, y);
        }
      });
    };
    drawColumn(firstColumn, 0);
    if (secondColumn.length) drawColumn(secondColumn, 1);

    if (footer) {
      ctx.fillStyle = '#475569';
      ctx.font = `600 ${Math.max(16, textFontSize - 4)}px ${textFontFamily}`;
      ctx.fillText(footer, padX, canvas.height - 62);
    }

    texture.needsUpdate = true;
  };

  draw({ title, lines });

  return {
    group,
    box,
    edges,
    width,
    height,
    depth,
    updateContent: draw,
  };
}

function createDecoderBottomVisual() {
  const group = new THREE.Group();
  group.name = 'decoderBottomVisual';
  architectureGroup.add(group);

  const createMiniPaperBoard = ({
    createMain,
    title,
    position,
    rotationZ = 0,
    scale = 0.22,
    ghostOffset = new THREE.Vector3(-26, 10, -22),
    ghostRotationZ = 0,
    showFaceBorder = false,
    showEdgeLines = false,
  }) => {
    const paper = createMain();
    const board = createPaperBoardMesh(paper, {
      title,
      showFaceBorder,
      showEdgeLines,
    });
    const ghost = createPaperBoardMesh(paper, {
      title,
      ghost: true,
      showFaceBorder,
      showEdgeLines,
    });
    ghost.group.position.copy(ghostOffset);
    ghost.group.rotation.z = ghostRotationZ;
    ghost.group.scale.setScalar(0.965);

    const wrapper = new THREE.Group();
    wrapper.position.copy(position);
    wrapper.rotation.z = rotationZ;
    wrapper.scale.setScalar(scale);
    wrapper.add(ghost.group);
    wrapper.add(board.group);
    group.add(wrapper);

    const refreshVisual = () => {
      const frontMaterial = Array.isArray(board.body.material) ? board.body.material[4] : null;
      const backMaterial = Array.isArray(board.body.material) ? board.body.material[5] : null;
      const nextMap = createSurveyBoardTexture(paper, { showBorder: showFaceBorder });
      if (frontMaterial?.map) frontMaterial.map.dispose();
      if (backMaterial?.map) backMaterial.map.dispose();
      if (frontMaterial) {
        frontMaterial.map = nextMap;
        frontMaterial.needsUpdate = true;
      }
      if (backMaterial) {
        backMaterial.map = nextMap.clone();
        backMaterial.needsUpdate = true;
      }
    };
    refreshVisual();

    board.body.userData.boxId = paper.kind === 'trip' ? 'tripPaper' : 'spPaper';
    moduleClickTargets.push(board.body);

    return {
      ...paper,
      group: wrapper,
      boardGroup: board.group,
      boardBody: board.body,
      boardHighlight: board.highlight,
      ghostGroup: ghost.group,
      boardWidth: board.width,
      boardHeight: board.height,
      boardDepth: board.depth,
      refreshVisual,
      basePosition: position.clone(),
      baseRotationZ: rotationZ,
      ghostBaseRotationZ: ghostRotationZ,
      scale,
    };
  };

  const miniBoardScale = 0.22;
  const miniBoardWidth = 700 * miniBoardScale;
  const miniBoardHeight = 920 * miniBoardScale;
  const rpCenterX = 304;
  const spCenterX = 604;
  const boardsCenterY = -646;
  const generateModuleCenterX = (rpCenterX + spCenterX) / 2;
  const generateModuleWidth = Math.abs(spCenterX - rpCenterX);
  const generateModuleHeight = miniBoardHeight * 0.5;
  const generateModuleCenterY = boardsCenterY - miniBoardHeight * 0.5 - 18 - generateModuleHeight * 0.5;
  const generateModuleDepthZ = 12;
  const xPanelCenterY = -452;
  const xOutputBusY = -392;

  const xPanels = [
    {
      id: 'x_rp',
      text: 'X_rp',
      x: 214,
      fill: '#d9ecff',
      width: 138,
      height: 74,
      cubeColor: '#b9dff5',
      shapeFormula: 'X_{rp} \\in \\mathbb{R}^{B\\times d_{rp}}',
      inputFormula: 'X_{rp} \\in \\mathbb{R}^{B\\times d_{rp}}',
      outputFormula: 'E_{rp} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}',
      tokenLabels: ['age', 'gender', 'job', 'car'],
    },
    {
      id: 'x_spCtx',
      text: 'X_hist',
      x: 374,
      fill: '#f8ebce',
      width: 148,
      height: 74,
      cubeColor: '#fcd86c',
      shapeFormula: 'X_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{hist}}',
      inputFormula: 'X_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{hist}}',
      outputFormula: 'E_{hist} \\in \\mathbb{R}^{B\\times L_{hist}\\times d_{model}}',
      tokenLabels: ['q_1', 'q_2', 'q_3', 'q_4'],
      historical: true,
    },
    {
      id: 'x_env',
      text: 'X_env',
      x: 534,
      fill: '#dcfce7',
      width: 138,
      height: 74,
      cubeColor: '#b7d59b',
      shapeFormula: 'X_{env} \\in \\mathbb{R}^{B\\times d_{env}}',
      inputFormula: 'X_{env} \\in \\mathbb{R}^{B\\times d_{env}}',
      outputFormula: 'E_{env} \\in \\mathbb{R}^{B\\times 1\\times d_{model}}',
      tokenLabels: ['zone', 'peak', 'weather', 'time'],
    },
    {
      id: 'x_cand',
      text: 'X_cand',
      x: 694,
      fill: '#ede9fe',
      width: 148,
      height: 74,
      cubeColor: '#d3c7ff',
      shapeFormula: 'X_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{cand}}',
      inputFormula: 'X_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{cand}}',
      outputFormula: 'E_{cand} \\in \\mathbb{R}^{B\\times L_{cand}\\times d_{model}}',
      tokenLabels: ['c_1', 'c_2', 'c_3', 'c_4'],
      historical: true,
    },
  ].map((item) => {
    const panel = createMiniVariableTokenPanel({
      title: item.text,
      shapeFormula: item.shapeFormula,
      inputFormula: item.inputFormula,
      outputFormula: item.outputFormula,
      width: item.width,
      height: item.height,
      depth: 10,
      fill: item.fill,
      border: '#cbd5e1',
      cubeColor: item.cubeColor,
      tokenLabels: item.tokenLabels,
      historical: item.historical,
    });
    panel.group.position.set(item.x, xPanelCenterY, 14);
    panel.box.userData.boxId = item.id;
    moduleClickTargets.push(panel.box);
    group.add(panel.group);
    return {
      ...item,
      panel,
      basePosition: panel.group.position.clone(),
    };
  });

  const rpPart = createMiniPaperBoard({
    createMain: createTripDiaryPaper,
    title: 'RP part · Trip Diary',
    position: new THREE.Vector3(rpCenterX, boardsCenterY, 10),
    rotationZ: 0,
    scale: miniBoardScale,
  });

  const spPart = createMiniPaperBoard({
    createMain: createSpSurveyPaper,
    title: 'SP part · Survey',
    position: new THREE.Vector3(spCenterX, boardsCenterY, 10),
    rotationZ: 0,
    scale: miniBoardScale,
  });

  const generateTitle = createOverlaySprite('Generate Questions', {
    width: 240,
    height: 36,
    fontSize: 18,
    scale: 0.1,
    fill: '#f8fbff',
    background: 'rgba(34,21,63,0.82)',
    border: 'rgba(226,232,240,0.18)',
  });
  generateTitle.position.set(generateModuleCenterX, generateModuleCenterY + generateModuleHeight * 0.5 + 32, 18);
  generateTitle.basePosition = generateTitle.position.clone();
  group.add(generateTitle);

  const generateFrame = new THREE.Mesh(
    new THREE.BoxGeometry(generateModuleWidth, generateModuleHeight, 4),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.02,
      depthWrite: false,
    }),
  );
  generateFrame.position.set(generateModuleCenterX, generateModuleCenterY, -6);
  group.add(generateFrame);

  const generateFrameEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(generateModuleWidth, generateModuleHeight, 4)),
    new THREE.LineDashedMaterial({
      color: '#cbd5e1',
      transparent: true,
      opacity: 0.42,
      dashSize: 12,
      gapSize: 8,
    }),
  );
  generateFrameEdges.computeLineDistances();
  generateFrameEdges.position.copy(generateFrame.position).add(new THREE.Vector3(0, 0, 1.4));
  group.add(generateFrameEdges);

  const generateConsole = createDynamicTextBoard({
    title: 'Generate Questions',
    width: generateModuleWidth - 10,
    height: generateModuleHeight - 10,
    depth: 8,
    fill: '#22153f',
    border: 'rgba(226,232,240,0.82)',
    titleColor: '#f8fbff',
    textColor: '#dbe4f5',
    accentColor: '#7ee787',
    titleFontSize: 28,
    textFontSize: 20,
    numbered: false,
    maxLinesPerColumn: 8,
    titleFontFamily: '"Avenir Next", "PingFang SC", Arial, sans-serif',
    textFontFamily: '"PingFang SC", Arial, sans-serif',
    resolutionScale: 8.4,
    lines: [
      '1. RP: R-208 / 女 / 31岁 / 上班族',
      '2. 基线: 珠江新城→天河智慧城 / 08:05-08:39 / 地铁',
      '3. Count Head: 本 block 生成 8 题（范围 8-12）',
      '4. Mask / Value Head: 并行输出 8 个题位的变量结构与取值',
      'C1 | 公交 15分 ¥2 | 小汽车 10分 ¥5',
      'C2 | 地铁 12分 ¥4 | 接驳步行 5分',
      'C3 | 公交候车 4分 | 拥挤度 中',
      'C4 | 停车 3分 | 停车费 ¥2',
      'C5 | 单车 18分 ¥1 | 舒适度 4/5',
      'C6 | 换乘 1次 | 准点率 88%',
      'C7 | 网约车 13分 ¥11 | 候车 3分',
      'C8 | 公交接驳 7分 | 步行 450m',
    ],
  });
  generateConsole.group.position.set(generateModuleCenterX, generateModuleCenterY - 2, generateModuleDepthZ);
  generateConsole.box.userData.boxId = 'outputBlock';
  moduleClickTargets.push(generateConsole.box);
  group.add(generateConsole.group);
  generateConsole.basePosition = generateConsole.group.position.clone();

  const blockFormula = createFormulaSprite('Q \\in \\mathbb{R}^{B\\times T_{block}\\times V}', {
    width: 620,
    height: 48,
    fontSize: 15,
    scale: 0.068,
    align: 'center',
    fill: '#fef3c7',
    background: 'rgba(0,0,0,0)',
    border: 'rgba(0,0,0,0)',
  });
  blockFormula.position.set(generateModuleCenterX, generateModuleCenterY + generateModuleHeight * 0.5 + 10, 16);
  blockFormula.basePosition = blockFormula.position.clone();
  group.add(blockFormula);

  const respondentProfiles = [
    {
      id: 'R-208',
      gender: '女',
      age: '31岁',
      job: '上班族',
      origin: '珠江新城',
      destination: '天河智慧城',
      depart: '08:05',
      arrive: '08:39',
      mode: '地铁',
      escort: '否',
    },
    {
      id: 'R-314',
      gender: '男',
      age: '42岁',
      job: '上班族',
      origin: '员村',
      destination: '琶洲',
      depart: '07:48',
      arrive: '08:18',
      mode: '公交',
      escort: '否',
    },
    {
      id: 'R-427',
      gender: '女',
      age: '27岁',
      job: '研究生',
      origin: '大学城',
      destination: '体育西路',
      depart: '09:12',
      arrive: '09:52',
      mode: '地铁',
      escort: '否',
    },
    {
      id: 'R-518',
      gender: '男',
      age: '38岁',
      job: '自由职业',
      origin: '番禺广场',
      destination: '客村',
      depart: '10:10',
      arrive: '10:46',
      mode: '小汽车',
      escort: '是',
    },
  ];

  const generatedScenarios = [
    {
      tag: 'DG8',
      subtitle: 'Decoder 生成的 8 行题面示例，突出时间、费用与接驳条件。',
      footer: '8 lines · values regenerated by decoder',
      combos: [
        'C1 | 公交 15分 ¥2 | 小汽车 10分 ¥5',
        'C2 | 地铁 12分 ¥4 | 接驳步行 5分',
        'C3 | 公交候车 4分 | 拥挤度 中',
        'C4 | 停车 3分 | 停车费 ¥2',
        'C5 | 单车 18分 ¥1 | 舒适度 4/5',
        'C6 | 换乘 1次 | 准点率 88%',
        'C7 | 网约车 13分 ¥11 | 候车 3分',
        'C8 | 公交接驳 7分 | 步行 450m',
      ],
      lines: [
        '公交出行时间 15 分钟，费用 2 元',
        '公交候车时间 4 分钟，拥挤度 中',
        '地铁出行时间 12 分钟，费用 4 元',
        '地铁换乘 1 次，步行接驳 5 分钟',
        '私家车出行时间 10 分钟，费用 5 元',
        '停车时间 3 分钟，停车费 2 元',
        '共享单车时间 18 分钟，费用 1 元',
        '舒适度评分 4 / 5',
      ],
      planA: ['公交时间 15 分钟', '费用 2 元', '候车 4 分钟'],
      planB: ['私家车时间 10 分钟', '费用 5 元', '停车 3 分钟'],
    },
    {
      tag: 'DG12',
      subtitle: 'Decoder 继续扩展为 12 行题面，加入舒适度、接驳与停车约束。',
      footer: '12 lines · values updated and pushed to SP part',
      combos: [
        'C1 | 公交 17分 ¥3 | 小汽车 11分 ¥6',
        'C2 | 公交候车 5分 | 拥挤度 高',
        'C3 | 地铁 13分 ¥4 | 换乘 1次',
        'C4 | 接驳步行 6分 | 舒适度 3/5',
        'C5 | 停车 4分 | 停车费 ¥3',
        'C6 | 网约车 12分 ¥11 | 候车 2分',
        'C7 | 单车 16分 ¥1 | 可靠性 84%',
        'C8 | 公交接驳 7分 | 步行 420m',
        'C9 | 座位率 40% | 准点率 82%',
        'C10 | 车内拥挤 低 | 空调 开',
        'C11 | time/cost 比值 | 1.18',
        'C12 | dominated flag | 0',
      ],
      lines: [
        '公交出行时间 17 分钟，费用 3 元',
        '公交候车时间 5 分钟，拥挤度 高',
        '地铁出行时间 13 分钟，费用 4 元',
        '地铁换乘 1 次，步行接驳 6 分钟',
        '私家车出行时间 11 分钟，费用 6 元',
        '停车时间 4 分钟，停车费 3 元',
        '网约车出行时间 12 分钟，费用 11 元',
        '共享单车时间 16 分钟，费用 1 元',
        '步行接驳时间 7 分钟，舒适度 3 / 5',
        '车内拥挤度 低，准点性 82%',
        '可用座位 40%，可靠性 88%',
        '方案差异阈值：time/cost 比 1.18',
      ],
      planA: ['公交时间 17 分钟', '费用 3 元', '候车 5 分钟'],
      planB: ['私家车时间 11 分钟', '费用 6 元', '停车 4 分钟'],
    },
  ];

  xPanels.forEach(({ panel }, index) => {
    createLiveArrow(() => {
      const start = getPanelWorldPoint(panel, 0, panel.height / 2 + 4, panel.depth * 0.5 + 2);
      const hub = group.localToWorld(new THREE.Vector3(generateModuleCenterX, xOutputBusY, 14));
      return [
        start,
        new THREE.Vector3(start.x, hub.y, start.z),
        hub,
      ];
    }, {
      color: '#f8fafc',
      opacity: 0.76,
      shaftWidth: 4.4,
      headWidth: 10,
      headLength: 12,
      depth: 3,
    });
  });

  [
    {
      from: rpPart,
      targets: [xPanels[0].panel, xPanels[2].panel],
    },
    {
      from: spPart,
      targets: [xPanels[1].panel, xPanels[3].panel],
    },
  ].forEach(({ from, targets }) => {
    targets.forEach((to) => {
      createLiveArrow(() => {
        const start = getPanelWorldPoint(from, 0, from.boardHeight * 0.56, from.boardDepth * 0.42);
        const end = getPanelWorldPoint(to, 0, -to.height / 2 - 4, to.depth * 0.5 + 2);
        const midY = start.y + (end.y - start.y) * 0.5;
        return [
          start,
          new THREE.Vector3(start.x, midY, start.z),
          new THREE.Vector3(end.x, midY, start.z),
          end,
        ];
      }, {
        color: '#cbd5e1',
        opacity: 0.74,
        shaftWidth: 4,
        headWidth: 10,
        headLength: 12,
        depth: 3,
      });
    });
  });

  createLiveArrow(() => {
    const start = getPanelWorldPoint(
      rpPart,
      -rpPart.boardWidth * 0.52,
      0,
      rpPart.boardDepth * 0.42,
    );
    const end = getPanelWorldPoint(
      generateConsole,
      -generateConsole.width / 2 - 4,
      -generateConsole.height * 0.12,
      generateConsole.depth * 0.5 + 3,
    );
    const laneX = Math.min(start.x, end.x) - 26;
    return [
      start,
      new THREE.Vector3(laneX, start.y, start.z),
      new THREE.Vector3(laneX, end.y, start.z),
      end,
    ];
  }, {
    color: '#61dafb',
    opacity: 0.82,
    shaftWidth: 4.8,
    headWidth: 11,
    headLength: 14,
    depth: 3,
  });

  createLiveArrow(() => {
    const start = moduleWorldLeftPointAt('maskHead', 0, 8, 18);
    const end = getPanelWorldPoint(
      generateConsole,
      -generateConsole.width / 2 - 4,
      generateConsole.height * 0.16,
      generateConsole.depth * 0.5 + 3,
    );
    const laneX = Math.min(start.x, end.x) - 172;
    return [
      start,
      new THREE.Vector3(laneX, start.y, start.z),
      new THREE.Vector3(laneX, end.y, start.z),
      end,
    ];
  }, {
    color: '#f97316',
    opacity: 0.82,
    shaftWidth: 4.8,
    headWidth: 11,
    headLength: 14,
    depth: 3,
  });

  createLiveArrow(() => {
    const start = moduleWorldRightPointAt('valueHead', 0, 8, 18);
    const end = getPanelWorldPoint(
      generateConsole,
      generateConsole.width / 2 + 4,
      generateConsole.height * 0.16,
      generateConsole.depth * 0.5 + 3,
    );
    const laneX = Math.max(start.x, end.x) + 176;
    return [
      start,
      new THREE.Vector3(laneX, start.y, start.z),
      new THREE.Vector3(laneX, end.y, start.z),
      end,
    ];
  }, {
    color: '#a78bfa',
    opacity: 0.8,
    shaftWidth: 4.6,
    headWidth: 10,
    headLength: 12,
    depth: 3,
  });

  createLiveArrow(() => {
    const start = getPanelWorldPoint(
      generateConsole,
      generateConsole.width / 2 + 4,
      -generateConsole.height * 0.12,
      generateConsole.depth * 0.5 + 3,
    );
    const end = getPanelWorldPoint(
      spPart,
      spPart.boardWidth * 0.52,
      0,
      spPart.boardDepth * 0.42,
    );
    const laneX = Math.max(start.x, end.x) + 58;
    return [
      start,
      new THREE.Vector3(laneX, start.y, start.z),
      new THREE.Vector3(laneX, end.y, start.z),
      end,
    ];
  }, {
    color: '#7ee787',
    opacity: 0.82,
    shaftWidth: 4.8,
    headWidth: 11,
    headLength: 14,
    depth: 3,
  });

  createLiveArrow(() => {
    const start = group.localToWorld(new THREE.Vector3(generateModuleCenterX, xOutputBusY, 14));
    const module = moduleMap.get('outputEmbedding');
    const end = getModuleWorldPoint(
      'outputEmbedding',
      0,
      -module.spec.size.y / 2 - 4,
      module.spec.depth * 0.5 + 2,
    );
    return [start, end];
  }, {
    color: '#f8fafc',
    opacity: 0.82,
    shaftWidth: 4.9,
    headWidth: 11,
    headLength: 14,
    depth: 3,
  });

  return {
    group,
    xPanels,
    rpPart,
    spPart,
    generateTitle,
    generateConsole,
    blockFormula,
    respondentProfiles,
    generatedScenarios,
    activeScenarioIndex: -1,
    activeScenarioTick: -1,
  };
}

const tripPaper = createPaperStack({
  createMain: createTripDiaryPaper,
  title: 'RP part · Trip Diary',
  position: new THREE.Vector3(-470, -760, 0),
  rotation: new THREE.Euler(0, 0, 0),
  ghostSign: -1,
  scale: 0.22,
});

const spPaper = createPaperStack({
  createMain: createSpSurveyPaper,
  title: 'SP part · Survey',
  position: new THREE.Vector3(-155, -760, 0),
  rotation: new THREE.Euler(0, 0, 0),
  ghostSign: 1,
  scale: 0.22,
});

const papers = { trip: tripPaper, sp: spPaper };
const decoderBottomVisual = createDecoderBottomVisual();
tripPaper.group.visible = true;
spPaper.group.visible = true;
tripPaper.ghosts.forEach((ghost) => { ghost.visible = true; });
spPaper.ghosts.forEach((ghost) => { ghost.visible = true; });

focusTargets.set('tripPaper', {
  position: new THREE.Vector3(-430, -620, 860),
  target: new THREE.Vector3(-470, -760, 0),
  label: 'Trip / RP 输入',
  kind: 'paper',
  paper: tripPaper,
});
focusTargets.set('spPaper', {
  position: new THREE.Vector3(-120, -620, 860),
  target: new THREE.Vector3(-155, -760, 0),
  label: 'SP Survey',
  kind: 'paper',
  paper: spPaper,
});
focusTargets.set('x_rp', {
  position: new THREE.Vector3(-700, 130, 860),
  target: new THREE.Vector3(-700, -504, 0),
  label: 'X_rp 输入特征',
  kind: 'overview',
});
focusTargets.set('x_spCtx', {
  position: new THREE.Vector3(-485, 130, 860),
  target: new THREE.Vector3(-485, -504, 0),
  label: 'X_hist 历史上下文',
  kind: 'overview',
});
focusTargets.set('x_env', {
  position: new THREE.Vector3(-245, 130, 860),
  target: new THREE.Vector3(-245, -504, 0),
  label: 'X_env 环境特征',
  kind: 'overview',
});
focusTargets.set('x_cand', {
  position: new THREE.Vector3(-25, 130, 860),
  target: new THREE.Vector3(-25, -504, 0),
  label: 'X_cand 候选模板',
  kind: 'overview',
});
focusTargets.set('architecture', {
  position: ARCHITECTURE_CAMERA.clone(),
  target: ARCHITECTURE_CENTER.clone(),
  label: '算法架构总览',
  kind: 'overview',
});
focusTargets.set('papers', {
  position: PAPER_OVERVIEW_CAMERA.clone(),
  target: SCENE_CENTER.clone(),
  label: '双页概览',
  kind: 'overview',
});

[
  ['concatPanel', 'Token Concat'],
  ['batchAttention', '多份数据自注意力'],
  ['maskHead', 'Mask Head'],
  ['valueHead', 'Value Head'],
  ['scoreHead', 'Score Head'],
  ['countHead', 'Count Head'],
  ['slotSelectHead', 'Slot Select Head'],
  ['outputBlock', '题组输出'],
  ['lossPanel', 'Loss 汇总'],
  ['backprop', '反向传播'],
  ['dispatchLoop', '问卷分发闭环'],
].forEach(([id, label]) => {
  focusTargets.set(id, {
    position: ARCHITECTURE_CAMERA.clone(),
    target: ARCHITECTURE_CENTER.clone(),
    label,
    kind: 'overview',
  });
});

[
  {
    from: () => getPaperTopCenter(tripPaper, 0),
    to: () => getModuleBottomCenter('x_rp', 8, 8),
    color: '#93c5fd',
  },
  {
    from: () => getPaperTopCenter(tripPaper, 0),
    to: () => getModuleBottomCenter('x_env', 8, 8),
    color: '#93c5fd',
  },
  {
    from: () => getPaperTopCenter(spPaper, 0),
    to: () => getModuleBottomCenter('x_spCtx', 8, 8),
    color: '#fbbf24',
  },
  {
    from: () => getPaperTopCenter(spPaper, 0),
    to: () => getModuleBottomCenter('x_cand', 8, 8),
    color: '#fbbf24',
  },
  {
    from: () => getModuleTopCenter('x_rp', 8, 8),
    to: () => {
      const module = moduleMap.get('inputEmbedding');
      return getModuleWorldPoint('inputEmbedding', (0.16 - 0.5) * module.spec.size.x, -module.spec.size.y / 2 - 10, 8);
    },
    color: '#e2e8f0',
    pathOptions: { sourceLift: 66, targetDrop: 10 },
  },
  {
    from: () => getModuleTopCenter('x_spCtx', 8, 8),
    to: () => {
      const module = moduleMap.get('inputEmbedding');
      return getModuleWorldPoint('inputEmbedding', (0.38 - 0.5) * module.spec.size.x, -module.spec.size.y / 2 - 10, 8);
    },
    color: '#e2e8f0',
    pathOptions: { sourceLift: 66, targetDrop: 10 },
  },
  {
    from: () => getModuleTopCenter('x_env', 8, 8),
    to: () => {
      const module = moduleMap.get('inputEmbedding');
      return getModuleWorldPoint('inputEmbedding', (0.62 - 0.5) * module.spec.size.x, -module.spec.size.y / 2 - 10, 8);
    },
    color: '#e2e8f0',
    pathOptions: { sourceLift: 66, targetDrop: 10 },
  },
  {
    from: () => getModuleTopCenter('x_cand', 8, 8),
    to: () => {
      const module = moduleMap.get('inputEmbedding');
      return getModuleWorldPoint('inputEmbedding', (0.84 - 0.5) * module.spec.size.x, -module.spec.size.y / 2 - 10, 8);
    },
    color: '#e2e8f0',
    pathOptions: { sourceLift: 66, targetDrop: 10 },
  },
  {
    from: () => getModuleTopCenter('inputEmbedding', 8, 8),
    to: () => getModuleBottomCenter('concatPanel', 8, 8),
    color: '#f8fafc',
    pathOptions: { sourceLift: 22, targetDrop: 8 },
  },
].forEach(({ from, to, color, pathOptions }) => {
  createLiveArrow(() => {
    const start = from();
    const end = to();
    return buildTopToBottomArrowPoints(start, end, pathOptions || {});
  }, { color, opacity: 0.74 });
});

function clearPaperFocus() {
  Object.values(papers).forEach((paper) => {
    if (paper.boardHighlight) paper.boardHighlight.material.opacity = 0;
    if (paper.boardBody) paper.boardBody.scale.set(1, 1, 1);
  });
}

function setPaperFocus(paper) {
  clearPaperFocus();
  if (paper && paper.boardHighlight) {
    paper.boardHighlight.material.opacity = 0.18;
    if (paper.boardBody) paper.boardBody.scale.set(1.02, 1.02, 1.02);
  }
}

function getModuleAnchor(id) {
  const module = moduleMap.get(id);
  return module ? module.group.position.clone() : new THREE.Vector3();
}

function getPaperAnchor(kind) {
  if (kind === 'tripPaper') {
    return tripPaper.basePosition.clone().add(new THREE.Vector3(84, 88, tripPaper.boardDepth));
  }
  return spPaper.basePosition.clone().add(new THREE.Vector3(-84, 88, spPaper.boardDepth));
}

createConnector('trip_to_input', 'tripPaper', 'inputEmbedding', '#61dafb', 120, 180);
createConnector('input_to_encoder', 'concatPanel', 'encoder', '#61dafb', 70, 0);
createConnector('encoder_to_output', 'encoder', 'decoderCrossAttention', '#8b9df7', 190, 140);
createConnector('output_to_qset', 'outputEmbedding', 'questionSetAttention', '#61dafb', 72, 0);
createConnector('qset_to_decoder', 'questionSetAttention', 'decoder', '#fbbf24', 120, 35);
createConnector('decoder_to_heads', 'decoderAddNormTop', 'scoreHead', '#fb7185', 82, 0);
createConnector('decoder_to_slot_select', 'decoderAddNormTop', 'slotSelectHead', '#fbbf24', 82, 0);
createConnector('heads_to_output', 'valueHead', 'outputBlock', '#7ee787', 82, 46);
createConnector('heads_to_weights', 'valueHead', 'weights', '#7ee787', 100, 120);
createConnector('weights_to_sp', 'weights', 'spPaper', '#fbbf24', 120, 120);
createConnector('sp_to_analysis', 'spPaper', 'analysis', '#61dafb', 120, -80);
createConnector('analysis_to_weights', 'analysis', 'weights', '#7ee787', 90, 0);
createConnector('weights_to_decoder', 'weights', 'decoder', '#fbbf24', 90, -120);

const routeOrb = new THREE.Mesh(
  new THREE.SphereGeometry(11, 24, 24),
  new THREE.MeshBasicMaterial({ color: 0x61dafb, transparent: true, opacity: 0.96 }),
);
routeOrb.visible = false;
routeOrb.material.opacity = 0;
webglScene.add(routeOrb);

function setModuleActive(id, active = true) {
  if (CANVAS_MODULE_IDS.has(id)) {
    if (active) diagramState.activeIds.add(id);
    else diagramState.activeIds.delete(id);
  }
  const module = moduleMap.get(id);
  if (!module) return;
  if (module.mesh) {
    const scale = active ? 1.035 : 1;
    module.mesh.scale.set(scale, scale, scale);
    module.group.position.z = active ? 12 : 0;
  }
  if (module.edgeLines && module.edgeLines.material) {
    module.edgeLines.material.opacity = active ? 1 : 0.96;
    module.edgeLines.material.color.set(active ? '#7dd3fc' : '#f8fafc');
  }
  if (module.dimensionSprite) {
    module.dimensionSprite.material.opacity = 0;
  }
}

function clearModuleHighlights() {
  diagramState.activeIds.clear();
  diagramState.activeConnectorId = '';
  diagramState.routeDuration = 0;
  moduleMap.forEach((_, id) => setModuleActive(id, false));
  connectorMap.forEach(({ material }) => {
    material.opacity = 0.18;
  });
}

function setActiveModuleButton(id) {
  moduleButtons.forEach((btn) => btn.classList.toggle('is-active', btn.dataset.target === id));
}

function tweenCamera(targetPosition, targetLookAt, duration = 820) {
  const fromPosition = camera.position.clone();
  const fromTarget = controls.target.clone();
  focusTween = {
    fromPosition,
    fromTarget,
    toPosition: targetPosition.clone(),
    toTarget: targetLookAt.clone(),
    duration,
    started: performance.now(),
  };
  return sleep(duration + 40);
}

async function focusTarget(id, { skipStatus = false } = {}) {
  const target = focusTargets.get(id);
  if (!target) return;
  clearModuleHighlights();
  clearPaperFocus();
  setActiveModuleButton(id);
  dashboardState.focusLabel = target.label;
  if (ENCODER_CANVAS_IDS.has(id)) setDiagramViewMode('encoder');
  if (DECODER_CANVAS_IDS.has(id)) setDiagramViewMode('decoder');
  setDiagramSelection(id);
  if (target.kind === 'module') {
    setModuleActive(id, true);
  }
  if (target.kind === 'paper') {
    setPaperFocus(target.paper);
  }
  setDimensionState(id, 'default');
  if (!skipStatus) {
    setStatus('模块聚焦', `已聚焦到 ${target.label}。中央架构图和右侧维度面板会同步高亮当前模块。`);
  }
  await tweenCamera(target.position, target.target, 820);
}

function updateHeadOutputs(values) {
  dashboardState.headOutputs = values.slice();
  statsRefs.head1Value.textContent = values[0].toFixed(2);
  statsRefs.head2Value.textContent = values[1].toFixed(2);
  statsRefs.head3Value.textContent = values[2].toFixed(2);
  statsRefs.head4Value.textContent = values[3].toFixed(2);
}

function applySpScenario(tag, subtitle, planA, planB) {
  dashboardState.scenarioTag = tag;
  spPaper.refs.scenarioTag.textContent = `SP part · ${tag}`;
  spPaper.refs.scenarioSubtitle.textContent = subtitle;
  planA.forEach((text, idx) => {
    if (spPaper.refs.planAItems[idx]) spPaper.refs.planAItems[idx].textContent = text;
  });
  planB.forEach((text, idx) => {
    if (spPaper.refs.planBItems[idx]) spPaper.refs.planBItems[idx].textContent = text;
  });
  if (spPaper.refreshVisual) spPaper.refreshVisual();
  updateDashboard();
}

function applySpScenarioToPaper(paper, tag, subtitle, planA, planB) {
  if (!paper?.refs) return;
  paper.refs.scenarioTag.textContent = `SP part · ${tag}`;
  paper.refs.scenarioSubtitle.textContent = subtitle;
  planA.forEach((text, idx) => {
    if (paper.refs.planAItems[idx]) paper.refs.planAItems[idx].textContent = text;
  });
  planB.forEach((text, idx) => {
    if (paper.refs.planBItems[idx]) paper.refs.planBItems[idx].textContent = text;
  });
  if (paper.refreshVisual) paper.refreshVisual();
}

function applyTripProfileToPaper(paper, profile) {
  if (!paper?.rows || !profile) return;
  paper.rows.date.value = '2026-04-24';
  paper.rows.member.value = profile.id || '';
  paper.rows.origin.value = profile.origin || '';
  paper.rows.destination.value = profile.destination || '';
  paper.rows.depart.value = profile.depart || '';
  paper.rows.arrive.value = profile.arrive || '';
  paper.rows.modeBus.checked = profile.mode === '公交';
  paper.rows.modeMetro.checked = profile.mode === '地铁';
  paper.rows.modeCar.checked = profile.mode === '小汽车';
  paper.rows.modeWalk.checked = profile.mode === '步行 / 骑行';
  paper.rows.transfer.checked = profile.transfer === '是';
  paper.rows.escort.checked = profile.escort === '是';
  if (paper.refreshVisual) paper.refreshVisual();
}

const SP_SCENARIOS = {
  G0: {
    subtitle: '模拟一页 SP 题面，包括方案单选、原因勾选与补充说明。',
    planA: ['公交接驳 5 分钟', '车内时间 14 分钟', '费用 3 元'],
    planB: ['地铁接驳 7 分钟', '车内时间 11 分钟', '费用 4 元'],
  },
  G1: {
    subtitle: '预训练后首次生成的 SP 题面，方案差异更集中在稳定性与接驳时间。',
    planA: ['公交接驳 4 分钟', '车内时间 13 分钟', '费用 3 元'],
    planB: ['地铁接驳 6 分钟', '车内时间 10 分钟', '费用 4 元'],
  },
  G2: {
    subtitle: '在线回写后更新的 SP 题面，进一步强化准点性与舒适性差异。',
    planA: ['公交接驳 5 分钟', '车内时间 15 分钟', '费用 3 元'],
    planB: ['地铁接驳 5 分钟', '车内时间 9 分钟', '费用 4 元'],
  },
};

function parseTimeToMinutes(value) {
  const raw = String(value || '').trim();
  const match = raw.match(/^(\d{1,2}):(\d{2})$/);
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]);
}

function formatDuration(mins) {
  if (mins == null || Number.isNaN(mins)) return '-';
  return `${mins} 分钟`;
}

function getSelectedTripMode() {
  if (tripPaper.rows.modeMetro.checked) return '地铁';
  if (tripPaper.rows.modeBus.checked) return '公交';
  if (tripPaper.rows.modeCar.checked) return '小汽车';
  if (tripPaper.rows.modeWalk.checked) return '步行/骑行';
  return '-';
}

function getSelectedPlan() {
  if (spPaper.rows.planB.checked) return '方案 B';
  if (spPaper.rows.planA.checked) return '方案 A';
  return '-';
}

function getSelectedReasons() {
  const picked = [];
  if (spPaper.rows.reasonTime.checked) picked.push('准点');
  if (spPaper.rows.reasonComfort.checked) picked.push('舒适');
  if (spPaper.rows.reasonFare.checked) picked.push('费用');
  if (spPaper.rows.reasonTransfer.checked) picked.push('换乘');
  return picked;
}

function computeCounts() {
  const rpFilled = [tripPaper.rows.date, tripPaper.rows.member, tripPaper.rows.origin, tripPaper.rows.destination]
    .filter((el) => String(el.value || '').trim()).length;
  const tripExtra = [tripPaper.rows.depart, tripPaper.rows.arrive]
    .filter((el) => String(el.value || '').trim()).length
    + (getSelectedTripMode() !== '-' ? 1 : 0)
    + (tripPaper.rows.transfer.checked ? 1 : 0)
    + (tripPaper.rows.escort.checked ? 1 : 0);
  const spFilled = [spPaper.rows.spId].filter((el) => String(el.value || '').trim()).length
    + (getSelectedPlan() !== '-' ? 1 : 0)
    + getSelectedReasons().length
    + (String(spPaper.rows.note.value || '').trim() ? 1 : 0);
  return { rpFilled, tripExtra, spFilled, total: rpFilled + tripExtra + spFilled };
}

function updateDashboard() {
  const counts = computeCounts();
  const totalFields = 16;
  const overallPct = Math.round((counts.total / totalFields) * 100);
  const spPct = Math.round((counts.spFilled / 7) * 100);
  const depart = parseTimeToMinutes(tripPaper.rows.depart.value);
  const arrive = parseTimeToMinutes(tripPaper.rows.arrive.value);
  const duration = depart != null && arrive != null && arrive >= depart ? arrive - depart : null;
  const respondent = String(tripPaper.rows.member.value || spPaper.rows.spId.value || '').trim() || '-';
  const origin = String(tripPaper.rows.origin.value || '').trim();
  const destination = String(tripPaper.rows.destination.value || '').trim();
  const reasons = getSelectedReasons();

  statsRefs.overallProgressValue.textContent = `${overallPct}%`;
  statsRefs.overallProgressFill.style.width = `${overallPct}%`;
  statsRefs.focusValue.textContent = dashboardState.focusLabel;
  statsRefs.stageValue.textContent = dashboardState.stage;
  statsRefs.stackCountValue.textContent = `Trip ${dashboardState.tripStacks} / SP ${dashboardState.spStacks}`;
  statsRefs.filledCountValue.textContent = `${counts.total} / ${totalFields}`;
  statsRefs.filledBreakdownValue.textContent = `RP+Trip ${counts.rpFilled + counts.tripExtra} / SP ${counts.spFilled}`;
  statsRefs.respondentValue.textContent = respondent;
  statsRefs.odValue.textContent = origin && destination ? `${origin} -> ${destination}` : (origin || destination || '-');
  statsRefs.durationValue.textContent = formatDuration(duration);
  statsRefs.modeValue.textContent = getSelectedTripMode();
  statsRefs.flagsValue.textContent = `换乘 ${tripPaper.rows.transfer.checked ? '是' : '否'} / 接送 ${tripPaper.rows.escort.checked ? '是' : '否'}`;
  statsRefs.trainingPhaseValue.textContent = dashboardState.trainingPhase;
  statsRefs.pretrainEpochValue.textContent = String(dashboardState.pretrainEpoch);
  statsRefs.onlineUpdateValue.textContent = String(dashboardState.onlineUpdate);
  statsRefs.weightVersionValue.textContent = dashboardState.weightVersion;
  statsRefs.analysisValue.textContent = dashboardState.analysisNote;
  setMathBlock(statsRefs.dimensionModuleValue, dashboardState.dimensionInfo.module);
  statsRefs.dimensionStageValue.textContent = dashboardState.dimensionInfo.stage;
  if (statsRefs.dimensionSourceValue) {
    statsRefs.dimensionSourceValue.textContent = getDocSourceText(dashboardState.focusId);
  }
  setMathBlock(statsRefs.dimensionBeforeValue, dashboardState.dimensionInfo.before);
  setMathBlock(statsRefs.dimensionAfterValue, dashboardState.dimensionInfo.after);
  setMathBlock(statsRefs.dimensionDetailValue, dashboardState.dimensionInfo.detail);
  updateHeadOutputs(dashboardState.headOutputs);
  statsRefs.scenarioTagValue.textContent = dashboardState.scenarioTag;
  statsRefs.planValue.textContent = getSelectedPlan();
  statsRefs.reasonCountValue.textContent = String(reasons.length);
  statsRefs.reasonListValue.textContent = reasons.length ? reasons.join('、') : '尚未勾选';
  statsRefs.spProgressValue.textContent = `${spPct}%`;
  statsRefs.spProgressFill.style.width = `${spPct}%`;
  if (statsRefs.flowSummaryValue) {
    statsRefs.flowSummaryValue.textContent = dashboardState.flowSummary;
  }
  tripPaper.refreshVisual?.();
  spPaper.refreshVisual?.();
  updateLeftDocPanel();
}

function getRowContainer(element) {
  return element.closest('[data-field]') || element.closest('[data-choice]') || element;
}

function clearActive(root) {
  root.querySelectorAll('.is-active').forEach((el) => el.classList.remove('is-active'));
}

function clearDone(root) {
  root.querySelectorAll('.is-done').forEach((el) => el.classList.remove('is-done'));
}

function markDone(target) {
  const row = getRowContainer(target);
  row.classList.remove('is-done');
  void row.offsetWidth;
  row.classList.add('is-done');
}

async function movePen(paper, target, duration = 360) {
  const row = getRowContainer(target);
  clearActive(paper.root);
  row.classList.add('is-active');
  setPaperFocus(paper);
  paper.refreshVisual?.();
  await sleep(duration + 40);
}

async function typeInto(paper, input, text, step = 82) {
  await movePen(paper, input);
  input.value = '';
  updateDashboard();
  for (const ch of text) {
    input.value += ch;
    updateDashboard();
    await sleep(step);
  }
  markDone(input);
  updateDashboard();
  await sleep(140);
}

async function checkField(paper, input, checked = true) {
  await movePen(paper, input);
  input.checked = checked;
  markDone(input);
  updateDashboard();
  await sleep(180);
}

async function selectRadio(paper, input) {
  await movePen(paper, input);
  input.checked = true;
  markDone(input);
  updateDashboard();
  await sleep(180);
}

function resetPaper(paper) {
  clearActive(paper.root);
  clearDone(paper.root);
  paper.pen.classList.remove('is-visible');
  paper.pen.style.transform = 'translate(18px, 24px) scale(0.85)';
  paper.root.querySelectorAll('input[type="text"], textarea').forEach((el) => {
    el.value = '';
  });
  paper.root.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach((el) => {
    el.checked = false;
  });
}

function resetDashboardState() {
  dashboardState.focusId = 'papers';
  dashboardState.focusLabel = '双页概览';
  dashboardState.stage = '初始化';
  dashboardState.trainingPhase = 'Idle';
  dashboardState.pretrainEpoch = 0;
  dashboardState.onlineUpdate = 0;
  dashboardState.weightVersion = 'w0';
  dashboardState.analysisNote = '尚未开始';
  dashboardState.scenarioTag = 'G0';
  dashboardState.headOutputs = [0.25, 0.25, 0.25, 0.25];
  dashboardState.flowSummary = '从 respondent 的 RP / Trip 输入开始，依次查看 embedding、concat、多头注意力、loss 计算、问卷生成与下一位 respondent 的循环。';
  dashboardState.dimensionInfo = {
    module: '双页概览',
    stage: '初始化',
    before: '尚未开始',
    after: '点击模块或运行训练流程后显示',
    detail: '维度变化会随模块高亮与训练流程同步刷新。',
  };
}

function resetAll() {
  Object.values(papers).forEach(resetPaper);
  clearSelectedEmbeddingNode();
  hoveredEmbeddingNode = null;
  clearHoveredConcatCube();
  resetDashboardState();
  clearDiagramOverlay();
  applySpScenario('G0', SP_SCENARIOS.G0.subtitle, SP_SCENARIOS.G0.planA, SP_SCENARIOS.G0.planB);
  clearModuleHighlights();
  setDiagramSelection('');
  diagramState.hoverId = '';
  clearPaperFocus();
  setFlowModeButton(null);
  setActiveModuleButton('');
  routeAnimation = null;
  routeOrb.visible = false;
  camera.position.copy(PAPER_OVERVIEW_CAMERA);
  controls.target.copy(SCENE_CENTER);
  setStatus('等待开始', '当前处于初始化状态。点击中央模块或右侧流程按钮，可查看问卷页与算法模块的不同视角和流程。');
  setDimensionState('papers', 'default');
}

async function animateRoute(connectorId, {
  duration = 1000,
  color = null,
  stage = null,
  focus = null,
  dimensionState = null,
  analysisNote = null,
  flowSummary = null,
  status = null,
} = {}) {
  const connector = connectorMap.get(connectorId);
  clearModuleHighlights();
  clearPaperFocus();

  const focusId = focus || (connector && typeof connector.to === 'string' ? connector.to : null);
  if (focusId && moduleMap.has(focusId)) {
    setModuleActive(focusId, true);
  }
  if (focusId === 'tripPaper') setPaperFocus(tripPaper);
  if (focusId === 'spPaper') setPaperFocus(spPaper);
  if (focusId) {
    const target = focusTargets.get(focusId);
    if (target) {
      dashboardState.focusLabel = target.label;
    } else if (moduleMap.has(focusId)) {
      dashboardState.focusLabel = moduleMap.get(focusId).label;
    }
    setActiveModuleButton(focusId);
    setDiagramSelection(focusId);
    setDimensionState(focusId, dimensionState || 'default');
  }

  if (connector) {
    connector.material.opacity = 0.8;
    if (color) connector.material.color.set(color);
    routeOrb.material.color.set(color || connector.color);
    routeOrb.visible = true;
    routeAnimation = {
      curve: connector.curve,
      started: performance.now(),
      duration,
    };
  } else {
    routeAnimation = null;
    routeOrb.visible = false;
  }
  diagramState.activeConnectorId = connectorId;
  diagramState.routeStarted = performance.now();
  diagramState.routeDuration = duration;
  if (stage) dashboardState.stage = stage;
  if (analysisNote) dashboardState.analysisNote = analysisNote;
  if (flowSummary) setFlowSummary(flowSummary);
  if (status) setStatus(status.badge, status.text);
  updateDashboard();
  await sleep(duration + 60);
  routeAnimation = null;
  routeOrb.visible = false;
  diagramState.activeConnectorId = '';
  diagramState.routeDuration = 0;
  if (connector) {
    connector.material.opacity = 0.18;
    connector.material.color.set(connector.color);
  }
}

async function playTripDiarySequence({
  date = '2026-04-16',
  respondentId = 'R-203',
  origin = '天河智慧城',
  destination = '珠江新城',
  depart = '08:10',
  arrive = '08:48',
} = {}) {
  dashboardState.stage = 'RP 信息采集';
  dashboardState.focusLabel = 'Trip / RP 输入';
  updateDashboard();
  setStatus('Trip Diary 动画中', '正在模拟一次 Trip Diary 的 RP 基础信息录入、出行方式选择与勾选动作。');
  await focusTarget('tripPaper', { skipStatus: true });
  await typeInto(tripPaper, tripPaper.rows.date, date);
  await typeInto(tripPaper, tripPaper.rows.member, respondentId);
  await typeInto(tripPaper, tripPaper.rows.origin, origin);
  await typeInto(tripPaper, tripPaper.rows.destination, destination);
  await typeInto(tripPaper, tripPaper.rows.depart, depart);
  await typeInto(tripPaper, tripPaper.rows.arrive, arrive);
  await selectRadio(tripPaper, tripPaper.rows.modeMetro);
  await checkField(tripPaper, tripPaper.rows.transfer, true);
  await checkField(tripPaper, tripPaper.rows.escort, true);
  setDimensionState('tripPaper', 'filled');
}

async function playSpSurveySequence({
  respondentId = 'R-203',
  note = '更看重准点性和舒适度。',
} = {}) {
  dashboardState.stage = 'SP 填答';
  dashboardState.focusLabel = 'SP Survey';
  updateDashboard();
  setStatus('SP Survey 动画中', '正在模拟 SP 方案选择、原因勾选与补充说明输入，右侧统计同时刷新。');
  await focusTarget('spPaper', { skipStatus: true });
  await typeInto(spPaper, spPaper.rows.spId, respondentId);
  await selectRadio(spPaper, spPaper.rows.planB);
  await checkField(spPaper, spPaper.rows.reasonTime, true);
  await checkField(spPaper, spPaper.rows.reasonComfort, true);
  await typeInto(spPaper, spPaper.rows.note, note, 76);
  setDimensionState('spPaper', 'answered');
}

async function presentScene({
  badge = '流程演示',
  text = '',
  flowSummary = '',
  focus = 'architecture',
  dimensionId = focus,
  dimensionState = 'default',
  overlayMode = 'idle',
  overlayStep = 'idle',
  overlayExtras = {},
  routeId = '',
  duration = 960,
  color = null,
  analysisNote = null,
  trainingPhase = null,
  pretrainEpoch = null,
  onlineUpdate = null,
  weightVersion = null,
  headOutputs = null,
} = {}) {
  if (trainingPhase !== null) dashboardState.trainingPhase = trainingPhase;
  if (pretrainEpoch !== null) dashboardState.pretrainEpoch = pretrainEpoch;
  if (onlineUpdate !== null) dashboardState.onlineUpdate = onlineUpdate;
  if (weightVersion !== null) dashboardState.weightVersion = weightVersion;
  if (headOutputs) updateHeadOutputs(headOutputs);
  setDiagramOverlay(overlayMode, overlayStep, { duration, ...overlayExtras });

  if (routeId) {
    await animateRoute(routeId, {
      duration,
      color,
      focus,
      dimensionState: dimensionId === focus ? dimensionState : 'default',
      analysisNote,
      flowSummary,
      status: { badge, text },
    });
    if (dimensionId && dimensionId !== focus) {
      setDiagramSelection(dimensionId);
      setDimensionState(dimensionId, dimensionState);
      updateDashboard();
    }
  } else {
    if (focus) {
      await focusTarget(focus, { skipStatus: true });
    }
    if (dimensionId) {
      setDiagramSelection(dimensionId);
      setDimensionState(dimensionId, dimensionState);
    }
    if (analysisNote) dashboardState.analysisNote = analysisNote;
    if (flowSummary) setFlowSummary(flowSummary);
    setStatus(badge, text);
    updateDashboard();
    await sleep(duration);
  }
}

async function playDataFlowSequence() {
  dashboardState.trainingPhase = 'Data Flow';
  dashboardState.pretrainEpoch = 0;
  dashboardState.onlineUpdate = 0;
  dashboardState.weightVersion = 'w0';
  setFlowSummary('数据流转模式：从 respondent 的 RP / Trip 填写开始，沿着 X_rp、X_env、X_hist、X_cand -> embedding -> concat -> attention -> 多头输出的路径依次展开。对 respondent 端，整份 SP block 始终是一次性生成并一次性填写。');
  await playTripDiarySequence();

  await presentScene({
    badge: '数据流转',
    text: 'Trip / RP 原始字段已经收集完成，现在拆分为 respondent 特征、环境统计、SP 历史上下文和候选模板四类输入。',
    flowSummary: '步骤 1：原始问卷字段被整理为 X_rp、X_env、X_hist、X_cand 四类输入张量。',
    focus: 'inputEmbedding',
    dimensionId: 'inputEmbedding',
    dimensionState: 'project',
    overlayMode: 'dataFlow',
    overlayStep: 'rawInputs',
    routeId: 'trip_to_input',
    duration: 1200,
    color: '#61dafb',
    analysisNote: 'RP / Trip 来自 respondent 页面；统计特征与变量槽位来自配置与历史题组。',
  });

  await presentScene({
    badge: '数据流转',
    text: '四类输入先被投影到统一的 d_model 空间，然后按 token 顺序拼接成 encoder 的输入序列。',
    flowSummary: '步骤 2：embedding 后统一到 d_model，并拼接成 X_enc [B,L_enc,d_model]。',
    focus: 'architecture',
    dimensionId: 'concatPanel',
    overlayMode: 'dataFlow',
    overlayStep: 'concat',
    routeId: 'input_to_encoder',
    duration: 1100,
    color: '#8b9df7',
    analysisNote: 'concat 后的序列会同时包含 respondent、统计、变量槽位和上下文信息。',
  });

  await presentScene({
    badge: '数据流转',
    text: '进入 encoder 后，d_model 会被拆成 h 个 head，每个 head 各自计算 Q、K、V 和注意力分数矩阵。',
    flowSummary: '步骤 3：在 encoder 中完成 Q / K / V 切分与自注意力计算，再通过 Add & Norm 和前馈网络形成 H_enc。',
    focus: 'encoder',
    dimensionId: 'encoder',
    dimensionState: 'attn',
    overlayMode: 'attention',
    overlayStep: 'batch',
    duration: 1100,
    analysisNote: '这里是 docs 里 Q_e、K_e、V_e -> S_e -> A_e -> O_e -> H_enc 的主路径。',
  });

  await presentScene({
    badge: '数据流转',
    text: '自注意力输出先经过残差归一化，再经过前馈网络和第二次 Add & Norm，得到稳定的 encoder 记忆 H_enc。',
    flowSummary: '步骤 4：Add & Norm 只改数值分布不改张量形状，FFN 做逐 token 非线性变换。',
    focus: 'encoderAddNormBottom',
    dimensionId: 'encoderAddNormBottom',
    duration: 820,
    analysisNote: '残差支路保留原信息通路，LayerNorm 稳定训练。',
  });

  await presentScene({
    badge: '采样建议',
    text: 'H_enc 还会分出一支 respondent_target_head，用来估计接下来更值得补充的 RP cell，例如某些区、年龄段或教育分段。',
    flowSummary: '步骤 5：respondent_target_head 输出 p_sample [B,C_sample]，与总体目标缺口融合成采样建议。',
    focus: 'encoder',
    dimensionId: 'respondentTargetHead',
    duration: 900,
    color: '#7ee787',
    analysisNote: '这一路是样本覆盖控制，不是 SP 题目生成 head；它服务于 dashboard 和调查员调度。',
  });

  await presentScene({
    badge: '数据流转',
    text: 'Decoder 端先接收并行 question queries，再进入 question-set 自注意力与交叉注意力，读取 H_enc 中的 respondent 条件信息。这里的题位是整份问卷内部的并行槽位，不再表示右移序列 token。',
    flowSummary: '步骤 6：Embedding&Concat -> question-set self-attention -> cross-attention -> slot hidden states；整份 block 生成完成后再一次性发放。',
    focus: 'outputEmbedding',
    dimensionId: 'outputEmbedding',
    dimensionState: 'tokens',
    routeId: 'encoder_to_output',
    duration: 900,
    color: '#8b9df7',
    analysisNote: 'Q_slot 是当前 block 的并行题位 queries，不再是右移后的历史 token。',
  });

  await presentScene({
    badge: '数据流转',
    text: 'Question-set self-attention 让同一份问卷中的多个题位彼此交互，用来控制重复度、覆盖度和结构多样性；这里不再使用上三角因果遮罩。',
    flowSummary: '步骤 7：question-set self-attention 建模的是整份问卷内部各题位之间的关系，而不是 respondent 的逐题作答顺序。',
    focus: 'questionSetAttention',
    dimensionId: 'questionSetAttention',
    dimensionState: 'masked',
    routeId: 'output_to_qset',
    duration: 900,
    color: '#61dafb',
    analysisNote: '这里保留的是 full self-attention 张量形状 [B,h,T_max,T_max]，不再额外施加上三角因果屏蔽。',
  });

  await presentScene({
    badge: '数据流转',
    text: '交叉注意力把并行题位 query 和 H_enc 的 key/value 融合，之后再经过 Add & Norm 与前馈网络，形成整份问卷各题位的隐藏状态。',
    flowSummary: '步骤 8：decoder 的输出 H_slot [B,T_max,d_model] 准备送入 count / slot_select / mask / value / score 多个输出头。',
    focus: 'decoderCrossAttention',
    dimensionId: 'decoderCrossAttention',
    routeId: 'qset_to_decoder',
    duration: 980,
    color: '#fbbf24',
    analysisNote: 'cross-attention 是把 respondent 条件、环境统计和历史问卷上下文真正注入各个题位表示的关键步骤。',
  });

  await presentScene({
    badge: '数据流转',
    text: '现在从并行题位表示分叉出四个输出头。先看 mask head，它负责决定每道题里哪些变量被激活。',
    flowSummary: '步骤 9：从并行题位表示分叉到 count / slot_select / mask / value / score 五个 head。',
    focus: 'maskHead',
    dimensionId: 'maskHead',
    routeId: 'decoder_to_mask',
    duration: 900,
    color: '#f97316',
    overlayMode: 'dataFlow',
    overlayStep: 'headDetail',
    overlayExtras: { headFocus: 'maskHead' },
    analysisNote: 'mask_head 输出 mask_logits [B,T_max,V] 与 mask_prob [B,T_max,V]。',
    headOutputs: [0.73, 0.44, 0.31, 0.18],
  });

  await presentScene({
    badge: '数据流转',
    text: 'value head 在 mask head 给出的结构条件下，为已经激活的变量生成具体值，既可以映射到离散水平，也可以保持连续值口径。',
    flowSummary: '步骤 10：value head 生成变量值，决定每道题里激活变量展示出来的具体 attribute value。',
    focus: 'valueHead',
    dimensionId: 'valueHead',
    routeId: 'decoder_to_value',
    duration: 900,
    color: '#14b8a6',
    overlayMode: 'dataFlow',
    overlayStep: 'headDetail',
    overlayExtras: { headFocus: 'valueHead' },
    analysisNote: 'value_head 对应 raw_value [B,T_max,V] -> value [B,T_max,V]，并与 variable mask 做 gating。',
    headOutputs: [0.73, 0.62, 0.31, 0.18],
  });

  await presentScene({
    badge: '数据流转',
    text: 'score head 给每个题位一个辅助质量分数，帮助系统评估整份问卷内部各题的区分度与信息量。',
    flowSummary: '步骤 11：score head 输出每个题位的质量分数，用于排序、约束和辅助奖励。',
    focus: 'scoreHead',
    dimensionId: 'scoreHead',
    routeId: 'decoder_to_score',
    duration: 900,
    color: '#a855f7',
    overlayMode: 'dataFlow',
    overlayStep: 'headDetail',
    overlayExtras: { headFocus: 'scoreHead' },
    analysisNote: 'score_head 输出 score [B,T_max,1]，不是最终题面本身，而是辅助估计量。',
    headOutputs: [0.73, 0.62, 0.58, 0.18],
  });

  await presentScene({
    badge: '数据流转',
    text: 'count head 只决定当前 respondent 这一份问卷生成多少题，不负责决定哪些题进入最终 block。',
    flowSummary: '步骤 12：count head 输出题数分布 p_count，并确定 T_q。',
    focus: 'countHead',
    dimensionId: 'countHead',
    routeId: 'decoder_to_count',
    duration: 900,
    color: '#ef4444',
    overlayMode: 'dataFlow',
    overlayStep: 'headDetail',
    overlayExtras: { headFocus: 'countHead' },
    analysisNote: 'count_head 输出 p_count [B,K_count]，再映射为最终题数 T_q 与 slot_select_head。',
    headOutputs: [0.73, 0.62, 0.58, 0.81],
  });

  await presentScene({
    badge: '数据流转',
    text: 'slot_select_head 随后对全部候选题位打分，并从 T_max 个 proposal slots 中选出 T_q 个进入最终问卷。这里没有“前 k 个默认有效”的时序假设。',
    flowSummary: '步骤 13：slot_select_head 输出 slot_logits [B,T_max] 与 M_slot [B,T_max]，完成 set/block 选择。',
    focus: 'slotSelectHead',
    dimensionId: 'slotSelectHead',
    routeId: 'decoder_to_slot_select',
    duration: 900,
    color: '#fbbf24',
    overlayMode: 'dataFlow',
    overlayStep: 'headDetail',
    overlayExtras: { headFocus: 'slotSelectHead' },
    analysisNote: 'M_slot 是集合选择结果，不是 attention mask，也不是默认按编号取前 k 个题位。',
    headOutputs: [0.73, 0.62, 0.58, 0.81],
  });

  await presentScene({
    badge: '数据流转完成',
    text: 'count / slot_select / mask / value / score 会被组装成真正的题组。题组写回 SP 页面后，就进入 respondent 作答与后续分析更新。',
    flowSummary: '步骤 14：count / slot_select / mask / value / score 被组装为 question block，并准备发放给 respondent。',
    focus: 'outputBlock',
    dimensionId: 'outputBlock',
    routeId: 'heads_to_output',
    duration: 980,
    color: '#7ee787',
    overlayMode: 'dispatch',
    overlayStep: 'count',
    overlayExtras: { countDecision: 'eight' },
    analysisNote: 'question block [B,T_q,V] 已生成，可进入问卷分发阶段。',
  });
}

async function playAttentionBatchSequence() {
  dashboardState.trainingPhase = 'Attention';
  dashboardState.pretrainEpoch = 0;
  dashboardState.onlineUpdate = 0;
  setFlowSummary('多份数据自注意力模式：强调 batch 内多个 respondent / block 同时进入 attention，展示 Q/K/V、分数矩阵 S、注意力权重 A 与 concat 后的 H。');
  await playTripDiarySequence();
  await focusTarget('architecture', { skipStatus: true });

  await presentScene({
    badge: '多份数据自注意力',
    text: '这里不再只看单个 respondent，而是把多份样本拼成 batch。每份样本在 batch 维 B 上并行，但各自独立计算注意力。',
    flowSummary: '步骤 1：多份 respondent / block 并行进入 encoder，形成批量 attention 计算图。',
    focus: 'architecture',
    dimensionId: 'batchAttention',
    overlayMode: 'attention',
    overlayStep: 'batch',
    overlayExtras: { sampleCount: 4 },
    duration: 1200,
    analysisNote: '这里的关键维度是 B、L_enc、h、d_h。',
  });

  await presentScene({
    badge: '多份数据自注意力',
    text: '每个样本都会把 X_enc 切成 Q、K、V，再按 head 维拆成 [B,h,L_enc,d_h]。注意力分数矩阵 S 的形状是 [B,h,L_enc,L_enc]。',
    flowSummary: '步骤 2：Q / K / V 切分与分数矩阵 S = QK^T / sqrt(d_h)。',
    focus: 'encoder',
    dimensionId: 'encoder',
    dimensionState: 'attn',
    routeId: 'input_to_encoder',
    duration: 1050,
    color: '#61dafb',
    overlayMode: 'attention',
    overlayStep: 'batch',
    overlayExtras: { sampleCount: 4 },
    analysisNote: '每个 head 都会各自输出一张注意力图，再在最后 concat 回 d_model。',
  });

  await presentScene({
    badge: '多份数据自注意力',
    text: '经过 softmax 之后得到注意力权重 A，随后用 O = A V 得到每个 head 的输出，再 concat 成 H_enc。',
    flowSummary: '步骤 3：从分数矩阵 S 得到注意力权重 A，再加权 Value 形成每个 head 的输出 O。',
    focus: 'batchAttention',
    dimensionId: 'batchAttention',
    duration: 1150,
    overlayMode: 'attention',
    overlayStep: 'batch',
    overlayExtras: { sampleCount: 4 },
    analysisNote: '你在文档里写的 H、S、Q/K/V 的维度关系，集中体现在这里。',
  });

  await presentScene({
    badge: '多份数据自注意力',
    text: 'Decoder 侧的 cross-attention 同样是按 batch 同时进行，只不过 query 来自 decoder，key/value 来自 encoder 的 H_enc。',
    flowSummary: '步骤 4：encoder 的批量记忆 H_enc 被 decoder 批量读取，形成条件化生成。',
    focus: 'decoderCrossAttention',
    dimensionId: 'decoderCrossAttention',
    routeId: 'qset_to_decoder',
    duration: 980,
    color: '#fbbf24',
    overlayMode: 'attention',
    overlayStep: 'batch',
    overlayExtras: { sampleCount: 4 },
    analysisNote: 'decoder cross-attention 的形状是 [B,h,T_max,L_enc] -> [B,T_max,d_model]。',
  });

  await presentScene({
    badge: '多份数据自注意力完成',
    text: '批量 attention 的结果最终回到多头输出层，由 count / slot_select / mask / value / score 五个 head 共同决定题数、变量结构、变量值和题组质量。',
    flowSummary: '步骤 5：批量 attention 只是计算机制，多头输出才负责真正的题组生成。',
    focus: 'maskHead',
    dimensionId: 'maskHead',
    routeId: 'decoder_to_mask',
    duration: 900,
    color: '#f97316',
    overlayMode: 'attention',
    overlayStep: 'batch',
    overlayExtras: { sampleCount: 4 },
    analysisNote: '注意力负责表示学习，多头负责题数 / 结构 / 数值 / 质量四类决策。',
  });
}

async function playTrainingProcessSequence() {
  dashboardState.trainingPhase = 'Train';
  dashboardState.pretrainEpoch = 1;
  dashboardState.onlineUpdate = 0;
  dashboardState.weightVersion = 'w0';
  setFlowSummary('训练模式：展示四个输出头各自的损失函数、总损失汇总、反向传播路径和参数更新。');
  await playTripDiarySequence();
  await focusTarget('architecture', { skipStatus: true });

  await presentScene({
    badge: '训练过程',
    text: '训练从多头输出开始监督。当前整份 block 的题位隐藏状态 H_slot 会并行送入 count、slot_select、mask、value、score 五个 head，分别得到题数、候选题选择、变量结构、变量值和质量预测。',
    flowSummary: '步骤 1：decoder 表示被送入五个 head，准备计算五类损失。',
    focus: 'maskHead',
    dimensionId: 'maskHead',
    routeId: 'decoder_to_mask',
    duration: 880,
    color: '#f97316',
    overlayMode: 'training',
    overlayStep: 'loss',
    analysisNote: 'L_count 通常用分类交叉熵；L_slot 可用 BCE / ranking / top-k matching；L_mask 用 BCE；L_value 用 masked regression；L_score 用 MSE。',
    headOutputs: [0.71, 0.58, 0.41, 0.22],
  });

  await presentScene({
    badge: '训练过程',
    text: '五个 head 的损失不会独立更新五套模型，而是按权重汇总成一个总损失，让公共的 encoder-decoder 主干一起更新。',
    flowSummary: '步骤 2：L_count、L_slot、L_mask、L_value、L_score 汇总成 L_total。',
    focus: 'lossPanel',
    dimensionId: 'lossPanel',
    duration: 1180,
    overlayMode: 'training',
    overlayStep: 'loss',
    analysisNote: 'L_total = λcLc + λmLm + λvLv + λqLq，必要时还可叠加约束惩罚和 reward 项。',
  });

  await presentScene({
    badge: '训练过程',
    text: '总损失确定后，梯度会从 head 反向流回 decoder，再回到 cross-attention、encoder 和 embedding，形成完整 backprop 链条。',
    flowSummary: '步骤 3：从多头损失出发，把梯度逆向传回 encoder-decoder 主干。',
    focus: 'backprop',
    dimensionId: 'backprop',
    duration: 1180,
    overlayMode: 'training',
    overlayStep: 'loss',
    analysisNote: '这里可对应你文档中的反向传播、critic / aux loss、以及参数约束项回传过程。',
  });

  dashboardState.pretrainEpoch = 2;
  dashboardState.weightVersion = 'w1';
  await presentScene({
    badge: '训练过程',
    text: '优化器执行一步参数更新后，新的权重会形成新的 checkpoint。之后 decoder 用新权重再生成题组，检查题数、变量结构和题面取值是否更合理。',
    flowSummary: '步骤 4：梯度更新 -> checkpoint refresh -> 新权重再次驱动题组生成。',
    focus: 'weights',
    dimensionId: 'weights',
    dimensionState: 'update',
    routeId: 'analysis_to_weights',
    duration: 1050,
    color: '#7ee787',
    overlayMode: 'training',
    overlayStep: 'loss',
    analysisNote: '参数更新结果最终固化为 θ_k -> θ_{k+1} / w_k -> w_{k+1}。',
    weightVersion: 'w1',
  });

  updateHeadOutputs([0.82, 0.66, 0.57, 0.36]);
  await presentScene({
    badge: '训练过程完成',
    text: '新权重会改变后续题组的题数、候选题选择、变量结构和数值。你现在看到的是训练后更稳定的一组多头输出。',
    flowSummary: '步骤 5：训练结束后，新的多头输出将进入问卷分发与在线更新闭环。',
    focus: 'outputBlock',
    dimensionId: 'outputBlock',
    routeId: 'heads_to_output',
    duration: 980,
    color: '#7ee787',
    overlayMode: 'training',
    overlayStep: 'loss',
    analysisNote: '训练的目的不是停在损失，而是改善后续题组生成和问卷分发质量。',
  });
}

async function playDispatchProcessSequence() {
  dashboardState.trainingPhase = 'Dispatch';
  dashboardState.pretrainEpoch = 2;
  dashboardState.onlineUpdate = 1;
  dashboardState.weightVersion = 'w1';
  applySpScenario('G1', SP_SCENARIOS.G1.subtitle, SP_SCENARIOS.G1.planA, SP_SCENARIOS.G1.planB);
  setFlowSummary('问卷分发模式：展示 count / slot_select / mask / value / score 共同生成题组、SP 页面发放、回收答案，再回到下一位 respondent 的 RP 填写。');
  await playTripDiarySequence();
  await focusTarget('architecture', { skipStatus: true });

  updateHeadOutputs([0.78, 0.65, 0.51, 0.22]);
  await presentScene({
    badge: '问卷分发',
    text: '系统先由 count_head 判定当前 respondent 这一份问卷需要生成 8 题。此时只确定题数，还没有决定具体是哪 8 个候选题。',
    flowSummary: '步骤 1：count_head 预测本份 block 的题数 T_q。',
    focus: 'countHead',
    dimensionId: 'countHead',
    routeId: 'decoder_to_count',
    duration: 980,
    color: '#ef4444',
    overlayMode: 'dispatch',
    overlayStep: 'count',
    overlayExtras: { countDecision: 'eight' },
    analysisNote: '这里不再是逐步终止概率判定，而是 count_head 对题数类别 K_count 做分类。',
  });

  await presentScene({
    badge: '问卷分发',
    text: 'slot_select_head 根据 slot_logits 在所有候选题位中选出 8 个进入最终 block。被选中的 slot 可再按展示规则排序，但排序不参与建模语义。',
    flowSummary: '步骤 2：slot_select_head 选择最终题集合，避免默认按编号取前 k 个题位。',
    focus: 'slotSelectHead',
    dimensionId: 'slotSelectHead',
    routeId: 'decoder_to_slot_select',
    duration: 980,
    color: '#fbbf24',
    overlayMode: 'dispatch',
    overlayStep: 'count',
    overlayExtras: { countDecision: 'eight' },
    analysisNote: 'M_slot ∈ {0,1}^{B×T_max} 是 set selection，不是题目时序 mask。',
  });

  updateHeadOutputs([0.78, 0.65, 0.51, 0.88]);
  await presentScene({
    badge: '问卷分发',
    text: '当题数、slot 选择、变量结构、变量取值都准备好后，系统就把整组题面一次性发放给 respondent。',
    flowSummary: '步骤 3：count / slot_select / mask / value / score 完成组装，当前 respondent 的题组定稿。',
    focus: 'outputBlock',
    dimensionId: 'outputBlock',
    routeId: 'heads_to_output',
    duration: 1000,
    color: '#7ee787',
    overlayMode: 'dispatch',
    overlayStep: 'count',
    overlayExtras: { countDecision: 'eight' },
    analysisNote: 'question block 已封装完成，即将写回 SP 页面。',
  });

  await presentScene({
    badge: '问卷分发',
    text: '题组写回 SP 页面后，respondent 会一次性收到这一整个 block，而不是一题一题在线等待。',
    flowSummary: '步骤 4：生成好的 block 被渲染到 SP Survey 页面，等待 respondent 一次性作答。',
    focus: 'spPaper',
    dimensionId: 'spPaper',
    dimensionState: 'generated',
    duration: 960,
    analysisNote: '分发时展示的是整组题，而不是单题实时弹出。',
  });

  await playSpSurveySequence({ respondentId: 'R-203', note: '当前 block 里第二题更能区分准点与舒适度。' });

  await presentScene({
    badge: '问卷分发',
    text: 'respondent 的作答、勾选原因和补充说明会一起回流到 analysis，用于后续重估与更新。',
    flowSummary: '步骤 5：SP 作答结果回收后进入 analysis，形成下一轮更新所需的统计与拟合信息。',
    focus: 'analysis',
    dimensionId: 'analysis',
    dimensionState: 'feedback',
    routeId: 'sp_to_analysis',
    duration: 980,
    color: '#61dafb',
    analysisNote: 'analysis 会整理回答、拟合指标、变量激活统计和敏感性信息。',
  });

  resetPaper(tripPaper);
  resetPaper(spPaper);
  applySpScenario('G0', SP_SCENARIOS.G0.subtitle, SP_SCENARIOS.G0.planA, SP_SCENARIOS.G0.planB);
  await presentScene({
    badge: '问卷分发',
    text: '当前 respondent 闭环结束后，系统回到下一位 respondent 的 RP / Trip 填写阶段，重新开始新的 block 生成。',
    flowSummary: '步骤 6：问卷分发并不是单次终止，而是回到新的 respondent，继续下一轮生成-分发-回收闭环。',
    focus: 'tripPaper',
    dimensionId: 'dispatchLoop',
    duration: 1000,
    overlayMode: 'dispatch',
    overlayStep: 'count',
    overlayExtras: { countDecision: 'eight' },
    analysisNote: '下一位 respondent 会继承最新权重，但拥有新的 RP / Trip 输入。',
  });
  await playTripDiarySequence({
    respondentId: 'R-204',
    origin: '琶洲西区',
    destination: '体育西路',
    depart: '08:24',
    arrive: '09:02',
  });
}

async function guardedPlay(playFn) {
  if (isPlaying) return;
  isPlaying = true;
  try {
    await playFn();
  } finally {
    isPlaying = false;
  }
}

playTripBtn.addEventListener('click', () => {
  resetAll();
  setFlowModeButton(playTripBtn);
  guardedPlay(playDataFlowSequence);
});

playSpBtn.addEventListener('click', () => {
  resetAll();
  setFlowModeButton(playSpBtn);
  guardedPlay(playAttentionBatchSequence);
});

playAllBtn.addEventListener('click', () => {
  resetAll();
  setFlowModeButton(playAllBtn);
  guardedPlay(playDispatchProcessSequence);
});

playTrainBtn.addEventListener('click', () => {
  resetAll();
  setFlowModeButton(playTrainBtn);
  guardedPlay(playTrainingProcessSequence);
});

focusArchitectureBtn.addEventListener('click', () => {
  if (isPlaying) return;
  focusTarget('architecture');
});

focusPapersBtn.addEventListener('click', () => {
  if (isPlaying) return;
  focusTarget('papers');
});

resetBtn.addEventListener('click', () => {
  if (isPlaying) return;
  resetAll();
});

toggleRotateBtn.addEventListener('click', () => {
  rotateEnabled = !rotateEnabled;
  controls.autoRotate = rotateEnabled;
  toggleRotateBtn.textContent = rotateEnabled ? '关闭自动旋转' : '开启自动旋转';
});

if (encoderTabBtn) {
  encoderTabBtn.addEventListener('click', () => {
    if (isPlaying) return;
    setDiagramViewMode('encoder');
  });
}

if (decoderTabBtn) {
  decoderTabBtn.addEventListener('click', () => {
    if (isPlaying) return;
    setDiagramViewMode('decoder');
  });
}

moduleButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    if (isPlaying) return;
    focusTarget(btn.dataset.target);
  });
});

if (diagramCanvas) {
  diagramCanvas.addEventListener('pointermove', (event) => {
    const hit = findDiagramHit(event.clientX, event.clientY);
    diagramState.hoverId = hit ? hit.id : '';
    diagramCanvas.style.cursor = hit ? 'pointer' : 'default';
  });

  diagramCanvas.addEventListener('pointerleave', () => {
    diagramState.hoverId = '';
    diagramCanvas.style.cursor = 'default';
  });

  diagramCanvas.addEventListener('pointerdown', (event) => {
    if (isPlaying) return;
    const hit = findDiagramHit(event.clientX, event.clientY);
    if (!hit) return;
    focusTarget(hit.id);
  });
}

webglRenderer.domElement.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  if (isPlaying) return;
  const {
    neuronHit, concatCubeHit, attentionTensorHit, moduleHit,
  } = getWebglIntersections(event);
  if (neuronHit && neuronHit.object) {
    clearHoveredConcatCube();
    clearHoveredAttentionTensorNode();
    focusTarget('inputEmbedding', { skipStatus: true });
    handleEmbeddingNodeSelection(neuronHit.object);
    return;
  }
  if (concatCubeHit && concatCubeHit.object) {
    if (hoveredEmbeddingNode) clearHoveredEmbeddingNode();
    clearHoveredAttentionTensorNode();
    focusTarget('concatPanel', { skipStatus: true });
    handleConcatCubeHover(concatCubeHit.object);
    moveEmbeddingTooltip(event);
    return;
  }
  if (attentionTensorHit && attentionTensorHit.object) {
    if (hoveredEmbeddingNode) clearHoveredEmbeddingNode();
    clearHoveredConcatCube();
    const info = attentionTensorHit.object.userData.attentionTensorNode;
    focusTarget(info.moduleId || 'encoder', { skipStatus: true });
    handleAttentionTensorHover(attentionTensorHit.object, { selected: true });
    moveEmbeddingTooltip(event);
    return;
  }
  if (!moduleHit) return;
  clearSelectedEmbeddingNode();
  clearHoveredEmbeddingNode();
  clearHoveredConcatCube();
  clearHoveredAttentionTensorNode();
  embeddingTooltip.style.display = 'none';
  const boxId = String(moduleHit.object.userData.boxId || '');
  if (!boxId) return;
  focusTarget(boxId);
});

webglRenderer.domElement.addEventListener('pointermove', (event) => {
  if (isPlaying) return;
  const {
    neuronHit, concatCubeHit, attentionTensorHit, moduleHit,
  } = getWebglIntersections(event);
  if (neuronHit && neuronHit.object) {
    clearHoveredConcatCube();
    clearHoveredAttentionTensorNode();
    handleEmbeddingNodeHover(neuronHit.object);
    moveEmbeddingTooltip(event);
  } else if (concatCubeHit && concatCubeHit.object) {
    if (hoveredEmbeddingNode) clearHoveredEmbeddingNode();
    clearHoveredAttentionTensorNode();
    handleConcatCubeHover(concatCubeHit.object);
    moveEmbeddingTooltip(event);
  } else if (attentionTensorHit && attentionTensorHit.object) {
    if (hoveredEmbeddingNode) clearHoveredEmbeddingNode();
    clearHoveredConcatCube();
    handleAttentionTensorHover(attentionTensorHit.object);
    moveEmbeddingTooltip(event);
  } else if (hoveredEmbeddingNode) {
    clearHoveredEmbeddingNode();
    if (!hoveredConcatCube) embeddingTooltip.style.display = 'none';
  } else if (hoveredConcatCube) {
    clearHoveredConcatCube();
    embeddingTooltip.style.display = 'none';
  } else if (hoveredAttentionTensorNode) {
    clearHoveredAttentionTensorNode();
    embeddingTooltip.style.display = 'none';
  } else {
    embeddingTooltip.style.display = 'none';
  }
  webglRenderer.domElement.style.cursor = (neuronHit || concatCubeHit || attentionTensorHit || moduleHit) ? 'pointer' : 'default';
});

webglRenderer.domElement.addEventListener('pointerleave', () => {
  if (isPlaying) return;
  webglRenderer.domElement.style.cursor = 'default';
  if (hoveredEmbeddingNode) clearHoveredEmbeddingNode();
  if (hoveredConcatCube) clearHoveredConcatCube();
  if (hoveredAttentionTensorNode) clearHoveredAttentionTensorNode();
  embeddingTooltip.style.display = 'none';
});

window.addEventListener('resize', () => {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  webglRenderer.setSize(width, height);
  renderDiagramCanvas();
});

resetAll();
syncDiagramTabs();
renderDiagramCanvas();

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const elapsed = clock.getElapsedTime();

  particles.rotation.y = elapsed * 0.03;
  particles.position.y = Math.sin(elapsed * 0.2) * 8;

  tripPaper.group.position.y = tripPaper.basePosition.y + Math.sin(elapsed * 0.62) * 5;
  tripPaper.group.rotation.z = tripPaper.baseRotation.z + Math.sin(elapsed * 0.34) * 0.006;
  spPaper.group.position.y = spPaper.basePosition.y + Math.sin(elapsed * 0.68 + 0.7) * 5;
  spPaper.group.rotation.z = spPaper.baseRotation.z + Math.sin(elapsed * 0.38 + 0.6) * 0.006;
  tripPaper.ghosts[0].rotation.z = -0.02 + Math.sin(elapsed * 0.33) * 0.005;
  tripPaper.ghosts[1].rotation.z = -0.012 + Math.sin(elapsed * 0.36 + 0.3) * 0.004;
  spPaper.ghosts[0].rotation.z = 0.02 + Math.sin(elapsed * 0.28 + 0.4) * 0.005;
  spPaper.ghosts[1].rotation.z = 0.012 + Math.sin(elapsed * 0.32 + 0.7) * 0.004;

  if (decoderBottomVisual) {
    decoderBottomVisual.rpPart.group.position.y = decoderBottomVisual.rpPart.basePosition.y + Math.sin(elapsed * 0.46 + 0.2) * 3.2;
    decoderBottomVisual.rpPart.group.rotation.z = decoderBottomVisual.rpPart.baseRotationZ;
    decoderBottomVisual.rpPart.ghostGroup.rotation.z = decoderBottomVisual.rpPart.ghostBaseRotationZ;

    decoderBottomVisual.spPart.group.position.y = decoderBottomVisual.spPart.basePosition.y + Math.sin(elapsed * 0.48 + 0.9) * 3.2;
    decoderBottomVisual.spPart.group.rotation.z = decoderBottomVisual.spPart.baseRotationZ;
    decoderBottomVisual.spPart.ghostGroup.rotation.z = decoderBottomVisual.spPart.ghostBaseRotationZ;

    decoderBottomVisual.generateConsole.group.position.y = decoderBottomVisual.generateConsole.basePosition.y
      + Math.sin(elapsed * 0.52 + 0.3) * 2.8;
    decoderBottomVisual.generateConsole.group.position.z = decoderBottomVisual.generateConsole.basePosition.z
      + Math.sin(elapsed * 0.94) * 1.2;
    decoderBottomVisual.generateTitle.position.y = decoderBottomVisual.generateTitle.basePosition.y
      + Math.sin(elapsed * 0.32) * 1.8;
    decoderBottomVisual.blockFormula.position.y = decoderBottomVisual.blockFormula.basePosition.y
      + Math.sin(elapsed * 0.42 + 0.2) * 1.2;

    decoderBottomVisual.xPanels.forEach(({ panel }, index) => {
      panel.group.position.y = decoderBottomVisual.xPanels[index].basePosition.y
        + Math.sin(elapsed * 0.38 + index * 0.42) * 2.2;
      if (typeof panel.animate === 'function') panel.animate(elapsed);
    });

    const scenarioCycle = elapsed / 3.2;
    const scenarioTick = Math.floor(scenarioCycle);
    const scenarioIndex = scenarioTick % decoderBottomVisual.generatedScenarios.length;
    if (!isPlaying && scenarioTick !== decoderBottomVisual.activeScenarioTick) {
      decoderBottomVisual.activeScenarioTick = scenarioTick;
      decoderBottomVisual.activeScenarioIndex = scenarioIndex;
      const currentScenario = decoderBottomVisual.generatedScenarios[scenarioIndex];
      const profile = decoderBottomVisual.respondentProfiles[
        scenarioTick % decoderBottomVisual.respondentProfiles.length
      ];
      const consoleLines = [
        `1. RP: ${profile.id} / ${profile.gender} / ${profile.age} / ${profile.job}`,
        `2. 基线: ${profile.origin}→${profile.destination} / ${profile.depart}-${profile.arrive} / ${profile.mode}`,
        `3. Count Head: 本 block 生成 ${currentScenario.lines.length} 题（范围 8-12）`,
        `4. Mask / Value Head: 并行输出 ${currentScenario.lines.length} 个题位的变量结构与取值`,
        ...(currentScenario.combos || currentScenario.lines.map((line, idx) => `C${idx + 1} ${line}`)),
      ];
      decoderBottomVisual.generateConsole.updateContent({
        title: `Generate Questions · ${currentScenario.tag}`,
        lines: consoleLines,
        footer: '流程：先接收 RP 与历史上下文，再由 Count / Mask / Value / Score Head 并行组装整份 SP block，随后一次性写入 SP 问卷',
      });
      applyTripProfileToPaper(decoderBottomVisual.rpPart, profile);
      applySpScenarioToPaper(
        decoderBottomVisual.spPart,
        currentScenario.tag,
        currentScenario.subtitle,
        currentScenario.planA,
        currentScenario.planB,
      );
      decoderBottomVisual.spPart.rows.spId.value = profile.id;
      decoderBottomVisual.spPart.rows.planA.checked = false;
      decoderBottomVisual.spPart.rows.planB.checked = true;
      decoderBottomVisual.spPart.rows.reasonTime.checked = true;
      decoderBottomVisual.spPart.rows.reasonComfort.checked = scenarioIndex === 0;
      decoderBottomVisual.spPart.rows.reasonFare.checked = scenarioIndex !== 0;
      decoderBottomVisual.spPart.rows.reasonTransfer.checked = true;
      decoderBottomVisual.spPart.rows.note.value = `基于 ${profile.origin}→${profile.destination} 的 RP 条件自动生成。`;
      if (decoderBottomVisual.spPart.refreshVisual) decoderBottomVisual.spPart.refreshVisual();
      applySpScenario(
        currentScenario.tag,
        currentScenario.subtitle,
        currentScenario.planA,
        currentScenario.planB,
      );
    }

    if (decoderBottomVisual.blockFormula?.material) {
      decoderBottomVisual.blockFormula.material.opacity = 0.72 + Math.sin(elapsed * 1.4) * 0.08;
    }
  }

  moduleMap.forEach((module) => {
    module.group.position.x = module.spec.position.x;
    module.group.position.y = module.spec.position.y + Math.sin(elapsed * 0.22 + module.spec.position.x * 0.0018) * 1.1;
    module.group.rotation.y = 0;
  });
  moduleAnimators.forEach((animateModule) => animateModule(elapsed));
  updateLiveArrows();

  if (focusTween) {
    const now = performance.now();
    const t = Math.min((now - focusTween.started) / focusTween.duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(focusTween.fromPosition, focusTween.toPosition, eased);
    controls.target.lerpVectors(focusTween.fromTarget, focusTween.toTarget, eased);
    if (t >= 1) focusTween = null;
  }

  if (routeAnimation) {
    const t = Math.min((performance.now() - routeAnimation.started) / routeAnimation.duration, 1);
    routeOrb.position.copy(routeAnimation.curve.getPointAt(t));
    if (t >= 1) routeAnimation = null;
  }

  controls.update();
  webglRenderer.render(webglScene, camera);
  renderDiagramCanvas();
}

animate();
