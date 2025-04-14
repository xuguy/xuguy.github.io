<!-- 页眉设置 -->
<div style="border-bottom: 1px solid #eaecef; padding-bottom: 0.5em; margin-bottom: 2em;">
<h7 align="center">MDS 5110 NLP Proposal for Final Project - Group 38: Guyuan XU 224040074

</div>

# Notes on Building a GPT from sctrach

## 1. Introduction  
This project aims at incorporating multiple well-reiceived tutorials (ref) to help real begginers gain detailed perspective on how a large language model works. We will build a standard GPT from sctrach along with rich explanation on every step and finally use it to to do downstream task like classification, following instruction and translation on a cpu-laptop. This tutorial will be available online.

**Highlights of tutorial structure:** (*Figure 1.9* from book [1](@ref))
<p align="center">
 <img src="mdfig\2025-04-11-14-11-43.png" width="70%" alt="描述"> 


## 2. Motivation  

### ***With so many useful tools and packages available, do we really need to understand their implementation details?***

It's important to note that I've attempted to select ​crucial implementation details​​ (e.g., text processing methods, structure of attention blocks, etc.) and explain their implementations. These details may not be exhaustive but are essential for building proper understanding. Our goal is to keep explanations ​**simple yet sufficiently informative​**​.

This approach is particularly helpful for **technical interviews you may encounter during job searches**, where demonstrating understanding beyond API calls is crucial.



### ***With so many textbooks available, what makes mine worth reading?***

In my journey of learning LLMs, I encountered numerous pitfalls in existing educational materials:

- **Most textbooks fall into two extremes**:
  1. Oversimplified content that only provides superficial understanding
  2. Excessively complex materials requiring substantial prerequisite knowledge

- ​**​Common shortcomings in "beginner-friendly" textbooks​**​:
  1. Omission of critical steps 
     - e.g., for example, what are those `__getitem__`, `.forward(...)` in code? How `Datasets` and `DataLoader` of `pytorch` works?
     - And many classic textbooks often omit data preprocessing steps, the reader may have no idea what shape of data are feed into models.
  2. Unexplained schematic diagrams (e.g., the classic Transformer architecture diagrams often confuse beginners, what do those arrows and line mean?) You gain **0** knowledge from the pictures<p align="center">
    <img src="mdfig\2025-04-11-14-27-43.png" width="30%" alt="描述"> 

  3. Implementation barriers using outdated or complex frameworks. I suppose you may not want to spend 95% of your time trying to re-build the experiment environment.

After extensively studying four well-plaused introductory textbooks [1,2,3,4](@ref) and several github repos [5](@ref), I've distilled their essence into this tutorial - genuinely demonstrating ​**​every step​**​ of model building and operation for absolute beginners, *without reservations*."

Besides, an important part of this tutorial is to introduce "**Best Practices** [6](@ref)" that are likely to be neglected. For example:

- Dealing with downloading: 1) Avoiding duplicate download by detetecting existed `data` file; 2) Retrying network connection/Constraining connection time
- Dealing with hyper-parameters: use `config` to manage hyperparameters
- Dealing with Errors: 1) Tensors in different devices; 2)CUDA out of mememorys


### ***​Why deviate from given directions to create a standalone project? What's its value?​**​*
   <!-- *(Alignment with course objectives, GitHub-driven vitality, suitability for true beginners while avoiding time sinks in trivial details)*   -->
Because I can't find a topic that fits mine:
   - Close to ​**TOPIC 10: A SURVEY FOR A SPECIFIC TOPIC​**​ (GPT implementation focus) but documents textbook-derived lessons:  
        > *Writing a survey paper involves a systematic review of relevant literature, critical analysis of methodologies and findings, and clear presentation of information accessible and insightful for researchers and practitioners.*
    
     - But here I review serveral materials I used for learning, my target is to clearly present all their benefits: some explain the model structure in great details, some present valuable intuition, some draw good pictures, etc.

   - Shares goals with ​**TOPIC 4: DATASET BUILDING​**​:  
     - **Aim to create lasting value for the beginners come after** (student of next term for example) rather than producing disposable coursework you and others might never see/use it again.

- ​Alignment with course objectives:​
I lack a solid foundation—weekly course content feels overwhelmingly complex. This project serves to lay a solid foundation for myself.


---

## 3. Expected Outcomes  
- Build a cpu-runnable GPT model steps by steps, exaplain very step and intuition for crusial step. 
- Apply self-made GPT model for downstream task. (classification, translation)
- Write a neat tutorial and put it online: https://xuguy.github.io/ (demo)
- **Keep updating these notes in the future after course ends**, I will add more AI related contents in the following studies.

## 8. References  
[1](@ref): Raschka S. (2024). *Build a Large Language Model (From Scratch)*  
[2](@ref): Koki Saitoh. (2016). *Deep Learning from the Basics: Python and Deep Learning: Theory and Implementation*
[3](@ref): 《深度学习入门2：自制框架》- 斋藤康毅；Koki Saitoh. (2020). *Deep Learning from Scratch 1* 
[4](@ref): 《数度学习进阶：自然语言处理》- 斋藤康毅；Koki Saitoh. (2018). *Deep Learning from Scratch 2: Natural Language Processing* 
[5](@ref): BabyGPT-Build_GPT_From_Scratch https://github.com/TatevKaren/BabyGPT-Build_GPT_From_Scratch
[6](@ref): Bret Slatkin. (2016) Effective Python: 90 Specific Ways to Write Better Python(2nd Edition)