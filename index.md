---
layout: default
title: 首页
---

# 欢迎来到 guobosun 的博客 👋

这里是我整理和分享 A/B 测试、数字营销 相关知识的地方。

📌 最新文章：

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small>（{{ post.date | date: "%Y-%m-%d" }}）</small>
    </li>
  {% endfor %}
</ul>

---

📚 页面导航：

- [关于我]({{ "/about.html" | relative_url }})
- [全部文章]({{ "/posts.html" | relative_url }}) *(可选，若你创建了 posts 页面)*

---

> Powered by GitHub Pages + Jekyll
