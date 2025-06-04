---
layout: post
title: Github page Tips
date: 2025-06-04
categories: Github
description: Learning
---
### Tips
1. guobosun.github.io（这是 GitHub Pages 的命名规范）
2. GitHub 的网页界面中，不能直接新建空文件夹，但可以通过新建一个文件的方式来“创建文件夹
3. GitHub 不允许直接删除整个空文件夹，因为 Git 是按内容跟踪的。你只能删除文件，当文件夹为空时，它会自动消失
4. Jekyll 规定博客文章必须放在 _posts/ 文件夹下，而且命名格式必须是：YYYY-MM-DD-title.md
5. css 文件夹（有时叫 assets/css/ 或 _sass/）在 Jekyll 博客中主要用于存放网站的样式文件
6. _sass/ 中的 *.scss 文件是模块化样式片段（不单独生成 CSS 文件）
7. /main.scss 是主入口文件，通过 @import 将 _sass/ 中的样式整合在一起，最终生成网站使用的 main.css
8. archive.html 代码是一个典型的 Jekyll 分类归档页面模板，它的功能是将博客文章按**分类（Category）**分组并展示，并在每篇文章后显示其发布日期
9. assets/ 是用于存放博客使用的静态资源，可以包含图片，CSS文件，JS文件等。
10. _includes/ 目录 是用于存放可复用的页面组件，比如页眉，页脚，导航链接，社交媒体链接等。
11. _layouts/ 目录 是用于定义页面的整体布局结构，包含default.html(主布局)以及post.html(文章布局) 等。
12. _config.yml 用于配置站点的基本信息和行为，以及使用模板，可以禁用远程主体：theme:null 从而使用自定义的_layouts/*.html。也支持配套主题：theme: minima 等。
13. <img src="http://ww3.sinaimg.cn/large/006tNc79gy1g5vv6nqdi5j30cc07oaa8.jpg" width="60%" alt="Mean_squared_error_loss" referrerPolicy="no-referrer"/>:referrerPolicy="no-referrer"` 是 HTML `<img>` 标签的一个属性，用于控制浏览器在请求图片时是否发送来源页面（referrer）信息
    <img src="/assets/lossfuction/Loss-fonction-01.jpg" width="60%" alt="Mean_squared_error_loss" /> 
