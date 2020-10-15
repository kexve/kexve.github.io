---
layout: post
title: jsDelivr学习
categories: 前端
---

## jsDelivr 支持 GitHub 资源的方式

### cdn 格式

> 用 S3 存储, 以确保 GitHub 故障或作者删除文件后, 所有文件依然可用

加载任何 GitHub release, commit, 或 branch:

``` text
https://cdn.jsdelivr.net/gh/user/repo@version/file
```

例如: 可用当前的 GitHub 仓库为例

``` text
https://cdn.jsdelivr.net/gh/kexve/kexve.github.io@master/assets/css/globals/common.css
https://cdn.jsdelivr.net/gh/kexve/kexve.github.io@0.4/assets/css/globals/common.css
```

也可以这样

``` text
https://cdn.jsdelivr.net/gh/jquery/jquery@32b00373b3f42e5cdcb709df53f3b08b7184a944/dist/jquery.min.js
```

用版本号来代表时, 可以用大的版本, 而不必精确到特定版本, 如

``` text
https://cdn.jsdelivr.net/gh/jquery/jquery@3.2/dist/jquery.min.js
https://cdn.jsdelivr.net/gh/jquery/jquery@3/dist/jquery.min.js
```

> 使用这一特性, 如果最新的版本不可用, 将自动版本回退, 而不会直接 404 error. 

还可以通过 latest 或不写版本号, 来强制使用最新版本

``` text
https://cdn.jsdelivr.net/gh/jquery/jquery@latest/dist/jquery.min.js
https://cdn.jsdelivr.net/gh/jquery/jquery/dist/jquery.min.js
```

> 不推荐使用

对 js/css 文件, 可以使用".min"获得压缩的文件, 如

``` text
https://cdn.jsdelivr.net/gh/jquery/jquery@3.2.1/src/core.min.js
```

> 压缩大文件可能耗时, 但只要压缩过几次, jsDelivr 就会生成永久文件

可以用 combine 来合并多个文件, 如

``` text
https://cdn.jsdelivr.net/combine/url1,url2,url3
```

> 单个文件适用的功能, 在这里也适用, 和压缩一样, 可能耗时, 但几次之后会生成永久文件 

### cdn 缓存

可以访问 `https://purge.jsdelivr.net/gh/mzlogin/mzlogin.github.io@1.2.0/assets/js/main.js` 来清除指定文件的缓存; (将引用的 CDN 链接里的 `cdn` 改成 `purge` 就是了)

## 利用 jsDelivr 加速博客

### 修改内容

1. 在 _config.yml 文件中添加控制开关: 

    ```yaml
    # 对 css 和 js 资源的 cdn 加速配置
    cdn:
        jsdelivr:
            enabled: true
    ```

2. 修改 _layouts 里的文件, 用 `assets_base_url` 代表加载静态资源的根路径:

    {% raw %}
    ```liquid
    {% assign assets_base_url = site.url %}
    {% if site.cdn.jsdelivr.enabled %}
        {% assign assets_base_url = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: '@master' %}
    {% endif %}
    ```
    {% endraw %}
   
3. 修改以前直接用 {% raw %}`{{ site.url }}`{% endraw %} 拼接的静态资源引用链接，替换为 {% raw %}`{{ assets_base_url }}`{% endraw %}，比如 _includes/header.html 里：

    {% raw %}
    ```diff
    - <link rel="stylesheet" href="{{ site.url }}/assets/css/posts/index.css">
    + <link rel="stylesheet" href="{{ assets_base_url }}/assets/css/posts/index.css">
    ```
    {% endraw %}

### 刷新 cdn 问题

如果项目曾经打过 tag, 那么新增/修改静态资源后, 需要刷新 CDN 缓存的话, 需要打个新 tag; 

一般发生在修改了博客模板的 js/css 以后. 

删除了所有的 tag，这样以前的 release 就变成了 Draft，对外是不可见的

## 参考链接

- [使用 jsDelivr 免费加速 GitHub Pages 博客的静态资源](https://mazhuang.org/2020/05/01/cdn-for-github-pages/#先看效果)
- [jsDelivr 官网介绍](https://www.jsdelivr.com/features#gh)

