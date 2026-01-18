import { defineConfig } from 'vitepress-theme-async/config';

import markdownItKatex from 'markdown-it-katex';

const customElements = [
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'mscarries',
  'msgroup',
  'mstack',
  'mlongdiv',
  'msline',
  'mstack',
  'mspace',
  'msqrt',
  'msrow',
  'mstack',
  'mstack',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'math',
  'mi',
  'mn',
  'mo',
  'ms',
  'mspace',
  'mtext',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'msqrt',
  'mstyle',
  'mmultiscripts',
  'mover',
  'mprescripts',
  'msub',
  'msubsup',
  'msup',
  'munder',
  'munderover',
  'none',
  'maligngroup',
  'malignmark',
  'mtable',
  'mtd',
  'mtr',
  'mlongdiv',
  'mscarries',
  'mscarry',
  'msgroup',
  'msline',
  'msrow',
  'mstack',
  'maction',
  'semantics',
  'annotation',
  'annotation-xml'
]

export default defineConfig({
	srcDir: './',
	themeConfig: {
          author: "Ruby",
          page: {
          archives: '/archives',
          categories: '/categories',
          tags: '/tags',
        },
        user: {
            name: "Ruby",
            firstName: "Ruby",
            lastName: "Scarlet",
            email: "scarlet_7255@outlook.com",
            domain: "\u7AD9\u70B9\u57DF\u540D",
            describe: "\u7F51\u7AD9\u7B80\u4ECB\u3002",
            avatar: "/pics/misc/avatar.jpg",
            ruleText: "\u6682\u4E0D\u63A5\u53D7\u4E2A\u4EBA\u535A\u5BA2\u4EE5\u5916\u7684\u53CB\u94FE\u7533\u8BF7\uFF0C\u786E\u4FDD\u60A8\u7684\u7F51\u7AD9\u5185\u5BB9\u79EF\u6781\u5411\u4E0A\uFF0C\u6587\u7AE0\u81F3\u5C1130\u7BC7\uFF0C\u539F\u521B70%\u4EE5\u4E0A\uFF0C\u90E8\u7F72HTTPS\u3002"
          
          },
          banner: {
            type: "img",
            bannerTitle: "",
            bannerText: "欢迎来到我的小世界",
            position: "top",
            fit: "cover",
            bgurl: "/pics/misc/banner.jpg",
          },
          copyrightYear: void 0,
          liveTime: {
            enable: false,
            prefix: "footer.tips",
            startTime: "04/02/2025 16:00:00"
          },
          sidebar: {
            typedTextPrefix: "",
            typedText: ["庄生晓梦迷蝴蝶，望帝春心托杜鹃"],
            social: [{
              name: "Github",
              url: "https://github.com/RubyScarlet7255",
              icon: "<img src=\"./pics/social/github.png\" height=\"30px\"/ width=\"30px\"></svg>",
            }],
          },
      topBars: [ 
			{ title: '主页', url: '/' }, 
			{ title: '归档', url: '/archives' }, 
      { title: "分类",
        children: [
          {title: "categories", url: '/categories'},
          {title: "tags", url: '/tags'},
        ]
      }
		] 
    },
    base: "/",
    vite: {
		resolve: {
		alias: {
			'@src': '../../src' // 使用别名代替绝对路径
		}
		// fs: {
		//   allow: ['E:/Project/blog', 'E:/Project/blog/node_modules/.bin']
		// }
		},
    },
	markdown: {
      config: (md: any) => {
          md.use(markdownItKatex)
      }
    },
});
