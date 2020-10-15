---  
layout: post  
title: Markdown的一些技巧  
categories: 前端  
---  
*注：如下技巧大多是利用 Markdown 兼容部分 HTML 标签的特性来完成，不一定在所有网站和软件里都完全支持，主要以 GitHub 支持为准。*  
@@dt
## 在表格单元格里换行
@@ds
借助于 HTML 里的 `<br />` 实现。  
示例代码：  
```
| Header1 | Header2                          |
|---------|----------------------------------|
| item 1  | one<br />two<br />three |
```  
示例效果：  
| Header1 | Header2                          |  
| ------- | -------------------------------- |  
| item 1  | one<br />two<br />three |  
@@dc
@@dt
## 图文混排
@@ds
使用 `<img>` 标签来贴图，然后指定 `align` 属性。  
示例代码：  
```
<img align="right" src="https://s3.jpg.cm/2020/08/15/uP6YQ.png"/>
这是一个示例图片。
图片显示在 N 段文字的右边。
N 与图片高度有关。
刷屏行。
刷屏行。
到这里应该不会受影响了，本行应该延伸到了图片的正下方，所以我要足够长才能确保不同的屏幕下都看到效果。
```  
示例效果：  
<img align="right" src="https://s3.jpg.cm/2020/08/15/uP6YQ.png"/>  
这是一个示例图片。  
图片显示在 N 段文字的右边。  
N 与图片高度有关。  
刷屏行。  
刷屏行。  
到这里应该不会受影响了，本行应该延伸到了图片的正下方，所以我要足够长才能确保不同的屏幕下都看到效果。  
@@dc
@@dt
## 行首缩进
@@ds
直接在 Markdown 里用空格和 Tab 键缩进在渲染后会被忽略掉，需要借助 HTML 转义字符在行首添加空格来实现，`&ensp;` 代表半角空格，`&emsp;` 代表全角空格。  
示例代码：  
```
&emsp;&emsp;春天来了，又到了万物复苏的季节。
```  
示例效果：  
&emsp;&emsp;春天来了，又到了万物复苏的季节。  
@@dc
@@dt
## 参考
@@ds
* <https://raw.githubusercontent.com/matiassingers/awesome-readme/master/readme.md>  
* <https://www.zybuluo.com/songpfei/note/247346>  

@@dc
