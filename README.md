# 使用

文章放在`_posts`目录下，命名为`yyyy-MM-dd-xxxx.md`，内容格式如下

```yaml
---
layout: post
title: 标题
categories: [分类1, 分类2]
topmost: true ~~可选~~
extMath: true ~~可选~~
---
文章内容，Markdown格式
```

文章资源放在`posts`目录，如文章文件名是`2019-05-01-theme_usage.md`，则该篇文章的资源需要放在`posts/2019/05/01`下,在文章使用时直接引用即可。当然了，写作的时候会提示资源不存在忽略即可

```md
![这是图片](xxx.png)

[xxx.zip 下载](xxx.zip)
```
