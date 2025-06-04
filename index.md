---
layout: default
---

# 欢迎来到 guobosun 的博客 👋


## 📌 最新文章

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small>（{{ post.date | date: "%Y-%m-%d" }}）</small>
    </li>
  {% endfor %}
</ul>

---

## 📚 页面导航

- [关于我]({{ "/about.html" | relative_url }})
