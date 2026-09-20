import { defineConfig } from 'vitepress';
import katexModule from '@vscode/markdown-it-katex';

const chapters = [
  ['01-temperature', '01 跑通温控'], ['02-plant', '02 看懂对象'],
  ['03-classical', '03 经典整定'], ['04-bo', '04 BO：让试验引导搜索'],
  ['05-fnn', '05 FNN：从案例学习'], ['06-qlearning', '06 Q-Learning：在线调参'],
  ['07-dqn', '07 DQN：用网络估计动作价值'], ['08-llm', '08 LLM：读反馈，再试一次'],
  ['09-comparison', '09 换个工况再比较'],
];

export default defineConfig({
  lang: 'zh-CN', title: '动手做 AI 整定',
  description: '同一个温控对象，七种选择 PI 参数的方法。运行、观察、修改，再解释。',
  base: '/hvac-ai-pid-demo/', cleanUrls: true,
  srcExclude: ['generated/**'],
  markdown: { config(md) { md.use(katexModule.default ?? katexModule, { throwOnError: true }); } },
  themeConfig: {
    nav: [{ text: '开始实验', link: '/chapters/01-temperature' }, { text: '方法比较', link: '/chapters/09-comparison' }],
    sidebar: [{ text: '实验课', items: chapters.map(([slug, text]) => ({ text, link: `/chapters/${slug}` })) }],
    outline: { level: [2, 3], label: '本页内容' },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/HlinForest/hvac-ai-pid-demo' }],
    docFooter: { prev: '上一个实验', next: '下一个实验' },
    footer: { message: '一个温度状态，一套 PI 控制器。每条曲线都来自可以重跑的实验。' },
  },
});
