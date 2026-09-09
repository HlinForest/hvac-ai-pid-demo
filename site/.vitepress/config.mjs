import { defineConfig } from "vitepress";

const chapters = [
  ["00", "我们要控制的，究竟是什么？"],
  ["01", "先让房间凉下来：跑通第一个温控实验"],
  ["02", "搭一间虚拟机房：热模型、压缩机与传感器"],
  ["03", "怎样比较才公平：建立我们的实验规则"],
  ["04", "Z-N：从一次阶跃试验算出 PI 参数"],
  ["05", "IMC：用一个 λ 调节快慢与稳健性"],
  ["06", "BO：让程序替我们寻找 Kp、Ki"],
  ["07", "Safe BO：不仅追求分数，也约束风险"],
  ["08", "FNN：让 PI 参数随着状态变化"],
  ["09", "RL：在虚拟房间里学习调参策略"],
  ["10", "LLM Agent：让模型提出建议，让程序负责验收"],
  ["11", "七种方法放到同一张考卷上"],
  ["12", "从 Python 到控制板：验证我们真的运行了同一套策略"],
];

export default defineConfig({
  lang: "zh-CN",
  title: "HVAC AI-PI 实验课",
  description: "面向初学者的机柜空调 PI 自动整定逐步实验报告",
  base: "/hvac-ai-pid-demo/",
  cleanUrls: true,
  lastUpdated: true,
  themeConfig: {
    logo: "/figures/fig1_系统结构框图.png",
    nav: [
      { text: "学习路线", link: "/" },
      { text: "主报告", link: "/main" },
      { text: "下载报告", link: "/downloads" },
    ],
    outline: { level: [2, 3], label: "本页目录" },
    sidebar: [
      { text: "从零开始", items: chapters.map(([id, text]) => ({ text: `${id} ${text}`, link: `/chapters/${id}` })) },
      { text: "工程文档", items: [{ text: "提交版主报告", link: "/main" }, { text: "复现与证据说明", link: "/evidence" }] },
    ],
    socialLinks: [{ icon: "github", link: "https://github.com/HlinForest/hvac-ai-pid-demo" }],
    search: { provider: "local" },
    footer: { message: "HVAC AI-PI 实验课 · 证据以仓库封存产物为准" },
  },
});
