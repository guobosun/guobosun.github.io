---
layout: post
title: "How to creative a io"
date: 2025-06-03
categories: Github
---
1. guobosun.github.io（这是 GitHub Pages 的命名规范）
2. GitHub 的网页界面中，不能直接新建空文件夹，但可以通过新建一个文件的方式来“创建文件夹
3. GitHub 不允许直接删除整个空文件夹，因为 Git 是按内容跟踪的。你只能删除文件，当文件夹为空时，它会自动消失

guobosun.github.io/
├── _config.yml            # 博客的主配置文件
├── _posts/                # 博客文章目录
│   └── 2025-06-03-ab-test.md
├── index.md               # 博客首页内容
├── about.md               # 关于页面
├── assets/                # 静态资源文件夹，如图片、CSS、JS
├── _layouts/              # 页面布局模板
├── _includes/             # 可复用组件，如导航栏、页脚等
├── _site/                 # 编译后的站点（本地使用，不上传）
└── README.md              # 仓库说明文件（不会展示在博客上）

  
| 文件/目录         | 类型   | 用途简介                                                                               |
| ------------- | ---- | ---------------------------------------------------------------------------------- |
| `_config.yml` | 配置文件 | 博客的**核心配置**文件，控制博客名称、主题、URL、插件等。                                                   |
| `_posts/`     | 文件夹  | 用于存放所有文章。每篇文章必须命名为 `YYYY-MM-DD-标题.md`，并包含 YAML 头部（Front Matter）。                   |
| `index.md`    | 页面文件 | 博客的主页内容，可自定义文本或自动列出文章列表。                                                           |
| `about.md`    | 页面文件 | 创建“关于我”页面，可在顶部导航中显示。                                                               |
| `assets/`     | 文件夹  | 存放博客中的图片、CSS、JS 等静态资源。可引用路径如：`/assets/img/logo.png`。                               |
| `_layouts/`   | 文件夹  | 存放网页的**页面结构模板**，例如 `default.html`、`post.html` 等。文章会套用这些布局。                         |
| `_includes/`  | 文件夹  | 可复用的页面片段，如导航栏 (`nav.html`)、页脚 (`footer.html`)。可在布局文件中 `{% include nav.html %}` 引用。 |
| `_site/`      | 自动生成 | Jekyll 编译后生成的最终网站目录，不需要上传。                                                         |
| `README.md`   | 说明文件 | 仓库说明（用于 GitHub 仓库首页，不展示在网站页面上）。                                                    |

✅ 1. _config.yml（核心配置）:控制博客样式、主题、地址等基本属性。
title: guobosun的博客
description: 分享 A/B 测试、数据分析等内容
theme: minima
url: "https://guobosun.github.io"
author: guobosun

✅ 2. _posts/2025-06-03-ab-test.md（博客文章）:所有文章必须保存在 _posts 文件夹，并带有上述 YAML 头信息。
---
layout: post
title: "A/B 测试简介"
date: 2025-06-03
categories: abtest
---
A/B 测试是一种常见的实验设计方法……

✅ 3. index.md（首页）
---
layout: home
---

欢迎访问我的博客！点击左侧导航浏览文章。

✅ 4. about.md（自定义页面）
---
layout: page
title: About
permalink: /about/
---

我是 guobosun，这是我的个人博客。


