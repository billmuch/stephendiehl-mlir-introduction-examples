# stephendiehl-mlir-introduction-examples

### 简介

偶然间读到Stephen Diehl关于MLIR的系列文章，其深入浅出的讲解令人印象深刻。

LLVM以及MLIR的出现，大大降低了编译器的技术门槛。但是其本身又是学习曲线非常陡峭一个工具。
大多数MLIR的Tutorial都是从如何创建一个新的Dialect，写一个转换Pass方面着手介绍，对入门者实在不够友好。

而此系列文章通过精心设计的与C/Python program直接交互的MLIR代码实例，对Tensor、Memref、Affine、Linalg等关键Dialect的核心概念进行剖析，通俗易懂的讲解了他们的功能特性。

特将此系列文章翻译成中文，并附上文中例子代码，希望能为相关领域的初学者和中级用户提供一些参考与帮助。

### 原文链接
https://www.stephendiehl.com/tags/compilers/

### 翻译链接


### 版权声明

本译文版权声明与原文保持一致，所有权利归属原作者。

All written content on this site is provided under a Creative Commons ShareAlike license. All code is provided under a MIT license unless otherwise stated.

### 作者简介
Stephen Diehl是一位居住在伦敦的软件程序员。https://www.stephendiehl.com/

他在Haskell语言，编译器和金融科技领域颇具影响力。

这里他的技术博客Posts以及一些文章和著作列表：

#### Python
* [Let's Write an LLVM Specializer for Python!](http://dev.stephendiehl.com/numpile/)

#### Haskell
* [Write You a Haskell](http://dev.stephendiehl.com/fun/)
* [Implementing a JIT Compiled Language with Haskell and LLVM](http://www.stephendiehl.com/llvm/)
* [What I Wish I Knew When Learning Haskell](http://dev.stephendiehl.com/hask)

#### Public Policy
* [Popping The Crypto Bubble](https://medium.com/@sdiehl/popping-the-crypto-bubble-99698f240b52)