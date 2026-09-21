import { defineConfig } from 'vitepress';
import katexModule from '@vscode/markdown-it-katex';

// Model downloads are assets, not VitePress page routes (also used by its client router).
process.env.VITE_EXTRA_EXTENSIONS = 'npz,pt';

const chapters = [
  ['01-temperature', '01 PI 控制执行'], ['02-plant', '02 对象与辨识'],
  ['03-classical', '03 经典整定'], ['04-bo', '04 BO：让试验引导搜索'],
  ['05-fnn', '05 FNN：从案例学习'], ['06-qlearning', '06 Q-Learning：在线调参'],
  ['07-dqn', '07 DQN：用网络估计动作价值'],
  ['10-ppo', '08 PPO：直接学习策略'], ['11-td3', '09 TD3：连续增益与双 Critic'],
  ['12-sac', '10 SAC：随机策略与熵'], ['13-crossq', '11 CrossQ：改变 Critic 的训练'],
  ['14-pg4pi', '12 PG4PI：PI 本身就是策略'],
  ['08-llm', '13 LLM：读反馈，再试一次'], ['09-comparison', '14 结果与成本'],
];

export default defineConfig({
  lang: 'zh-CN', title: 'AI 自动整定技术解析',
  description: '同一个温控对象，十二种选择 PI 参数的方法。逐步计算、实现、训练与验证。',
  base: '/hvac-ai-pid-demo/', cleanUrls: true,
  srcExclude: ['generated/**'],
  markdown: { config(md) { md.use(katexModule.default ?? katexModule, { throwOnError: true }); } },
  themeConfig: {
    nav: [{ text: '控制基础', link: '/chapters/01-temperature' }, { text: '方法比较', link: '/chapters/09-comparison' }],
    sidebar: [{ text: '技术解析', items: chapters.map(([slug, text]) => ({ text, link: `/chapters/${slug}` })) }],
    outline: { level: [2, 3], label: '本页内容' },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/HlinForest/hvac-ai-pid-demo' }],
    docFooter: { prev: '上一章', next: '下一章' },
    footer: { message: '一个温度状态，一套 PI 控制器。每条曲线都来自可以重跑的实验。' },
  },
});
