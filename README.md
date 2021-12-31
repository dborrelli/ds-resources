# Data Science Resources
This list started out as a way for me to keep track of data science resources I've found helpful. However, I frequently get asked for data science resource recommendations by other data scientists and friends looking to break into data science. So I've continued to add to this, with a focus on beginner- and intermediate-level resources. Where possible, I've included links to the (legitimate) free versions of books. One of the great things about the data science community is the willingness to open-source and make things available for free. Within each category or sub-category the resources are listed very loosely in order of usefulness/introductory level to more advanced (but not entirely).

This list is far from complete, but I'll try to continue to add to it. Hopefully you find it helpful.

Non-exhaustive list of additional topics to add:
 - Spark
 - Git/GitHub
 - time series forecasting
 - docker

<br><br>

## Technical Resources

### 1. Foundational
#### 1.1 Python
- `course` [Coursera - Introduction to Data Science in Python](https://www.coursera.org/learn/python-data-analysis)
- `course` [codecademy - Learn Python 3](https://www.codecademy.com/learn/learn-python-3)
- `ebook` [Python Like You Mean It](https://www.pythonlikeyoumeanit.com/)
- `book` [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/#toc)
- `book` [Python for Everybody](https://www.py4e.com/book.php)
- `book` [Learn Python the Hard Way](https://learnpythonthehardway.org/) maybe not my favorite resource, but was still useful
- `book` [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
- `video series` [Calm Code](https://calmcode.io/)
- [Google Python style guide](https://google.github.io/styleguide/pyguide.html)

#### 1.2 Statistics
- `course` [Khan Academy - Statistics](https://www.khanacademy.org/math/ap-statistics)
- [Stanford Experimental Design course](https://statweb.stanford.edu/~owen/courses/363/) and [course notes](https://statweb.stanford.edu/~owen/courses/363/doenotes.pdf)
- `course` [Coursera - Statistics with Python Specialization](https://www.coursera.org/specializations/statistics-with-python)
- `book` [Open Intro Statistics](https://www.openintro.org/book/os/)
- `book` [Introduction to Empirical Bayes](https://drob.gumroad.com/l/empirical-bayes) by David Robinson
- `book` [Think Bayes](http://allendowney.github.io/ThinkBayes2/index.html)
- `book` [Think Stats](https://greenteapress.com/thinkstats2/thinkstats2.pdf)

#### 1.3 SQL
- [Mode SQL tutorial](https://mode.com/sql-tutorial/introduction-to-sql/)
- [SQL Zoo](https://sqlzoo.net/wiki/SQL_Tutorial)

#### 1.4 Computer Science, data structures and algorithms
- `course` - `MIT OCW` - [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/index.htm)
- `ebook` [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds3/index.html)
- `course` [Khan Academy - Algorithms](https://www.khanacademy.org/computing/computer-science/algorithms)
- `course` [Coursera - Algorithms Specialization](https://www.coursera.org/specializations/algorithms#courses)
- `course` - `MIT OCW` - [Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/index.htm)
- [HackerRank 30 days of code](https://www.hackerrank.com/domains/tutorials/30-days-of-code)
- `github repo` [Awesome Algorithms](https://github.com/tayllan/awesome-algorithms#online-courses)
- `book` [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-third-edition) by Cormen, Leiserson, Rivest and Stein
- `article` [Learn X in Y minutes: Bash](https://learnxinyminutes.com/docs/bash/)
- `article` [Bash scripting cheatsheet](https://devhints.io/bash)



### 2. General ML
#### 2.1 ML overview
- `course` [Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning) foundational knowledge of machine learning
- `course` [Applied Data Science with Python Specialization](https://www.coursera.org/specializations/data-science-python) more immediately applicable than the previous course
- `book` [An Introduction to Statistical Learning with Applications in R (ISLR), 2nd edition](https://www.statlearning.com/) by James, Witten, Hastie, Tibshirani
- `book` [The Hundred Page Machine Learning book](http://themlbook.com/wiki/doku.php)
- `book` [Approaching (Almost) Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost)
- `book` [Mining of Massive Datasets](http://www.mmds.org/) and `course` [edX/Stanford - Mining Massive Datasets](https://www.edx.org/course/mining-massive-datasets)
- `book` (advanced material) [Probabilistic Machine Learning: An Introduction by Kevin Murphy](https://probml.github.io/pml-book/book1.html)
- `book` (advanced material) [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Papers With Code](https://paperswithcode.com/)
- [ArXiv Sanity Preserver](http://www.arxiv-sanity.com/)

#### 2.2 University ML courses
- `course` - `Harvard` [CS 109 Data Science](http://cs109.github.io/2015/pages/videos.html)
- `course` - `Cornell` CS 4780 Machine Learning [lecture notes](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) and [lecture youtube videos](https://www.youtube.com/playlist?list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS)
- `course` - `MIT` [Intro to Machine Learning](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/about)
- `course` - `Wisconsin` [Machine Learning](https://sebastianraschka.com/teaching/stat479-fs2019/) Sebastian Raschka


#### 2.3 Dimensionality reduction
- `paper` [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426.pdf) by McInnes et al.
- `blog` [Understanding UMAP](https://pair-code.github.io/understanding-umap/) by Andy Coenen and Adam Pearce
- `article` [How Exactly UMAP Works](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668)
- `paper` [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by van der Maaten and Hinton
- `blog` [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

#### 2.4 Clustering
- `blog` [Visualizing DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) by Naftali Harris
- `API documentation` [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- `paper` [Accelerated Hierarchical Density Clustering](https://arxiv.org/pdf/1705.07321.pdf) by McInnes and Healy, 2017
- `blog` [Understanding HDBSCAN and Density-Based Clustering](https://pberba.github.io/stats/2020/01/17/hdbscan/) by Pepe Berba
- `stackoverflow` [How to select a clustering method? How to validate a cluster solution?](https://stats.stackexchange.com/a/195481)
- `stackoverflow` [Evaluation measures of goodness or validity of clustering](https://stats.stackexchange.com/a/358937)
- `paper` [What are the true clusters?](https://arxiv.org/abs/1502.02555) by Christian Henning
- `paper` [Density-Based Clustering Validation](https://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96) by Moulavi et al, 2014

#### 2.5 Curse of dimensionality
- `paper` [On the Surprising Behavior of Distance Metrics
in High Dimensional Space](https://bib.dbvis.de/uploadedFiles/155.pdf) by Aggarwal et al., 2001
- `blog` [Escaping the Curse of Dimensionality](https://www.freecodecamp.org/news/the-curse-of-dimensionality-how-we-can-save-big-data-from-itself-d9fa0f872335/) by Peter Gleeson (FreeCodeCamp)

#### 2.6 Data issues
- `article` [Learning from Imbalanced Classes](https://www.svds.com/learning-imbalanced-classes/)

### 3. ML in production
- `github repo` [Curated papers, articles, and blogs on data science & machine learning in production](https://github.com/eugeneyan/applied-ml)
- `course` [Stanford CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html)
- `article` [Overview of the different approaches to putting Machine Learning (ML) models in production](https://medium.com/analytics-and-data/overview-of-the-different-approaches-to-putting-machinelearning-ml-models-in-production-c699b34abf86)
- `article` [A Practical Guide to Maintaining Machine Learning in Production](https://eugeneyan.com/writing/practical-guide-to-maintaining-machine-learning/)

### 4. MLOps
- `github repo and tutorials` [Made With ML](https://madewithml.com/) by Goku Mohandas
- `github repo` [Awesome ML Ops](https://github.com/visenger/awesome-mlops)
- `article` [ML Ops: Machine Learning as an Engineering Discipline](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f)

### 5. Deep Learning
#### 5.1 General DL
- `course` [Coursera - deeplearning.ai Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- `book` [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow-dp-1492032646/dp/1492032646/ref=dp_ob_title_bk) and its associated [github repo](https://github.com/ageron/handson-ml2) (the first ~200 pages are about general ML so this book could go under that section, but it's probably better suited for someone looking to learn about DL)
- `course` [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- `site` [Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.59180&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

#### 5.2 University DL courses
- `github repo` [Deep Learning Drizzle](https://github.com/kmario23/deep-learning-drizzle) giant list of university DL courses
- `course` [Stanford CS230 - Deep Learning](https://cs230.stanford.edu/)
- `course` [Stanford CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/index.html)
- `course` [Yann LeCun's NYU course - DS-GA 1008 · SPRING 2020](https://atcold.github.io/pytorch-Deep-Learning/)
- `course` [MIT Intro to Deep Learning](http://introtodeeplearning.com/)

#### 5.3 DL papers
- `github repo` [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
- `paper` [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) by He et al, 2015
- `paper` [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820) by Smith, 2018

#### 5.4 TensorFlow
- `book` [Deep Learning with Python, 2nd edition](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet
- `course` [Coursera Deeplearning.AI Tensorflow Developer Professional Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)

#### 5.5 PyTorch
- `book` [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

#### 5.6 Reinforcement Learning
- `course` [Coursera - Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning#courses)
- `book` [Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) by Sutton and Barto

#### 5.7 Graph Neural Networks
- `article` [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)

### 6. NLP
#### 6.1 NLP overview
- `course` [Coursera - deeplearning.ai Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearningai&utm_medium=institutions&utm_content=NLP_6/17_ppt#courses)
- `course` [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html)
- `course` [Advanced NLP with spaCy](https://course.spacy.io/en)
- `course` [Hugging Face course](https://huggingface.co/course/chapter1)
- `book` [Natural Language Processing with Python](https://www.nltk.org/book/) by Bird, Klein and Loper
- `course` [Michigan NLP course videos](https://www.youtube.com/channel/UCYGBs23woNtXUSl6AugHNXw) and [github](https://github.com/deskool/nlp-class)
- `article` [FROM Pre-trained Word Embeddings TO Pre-trained Language Models — Focus on BERT](https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598)
- `book` [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) by

#### 6.2 Embeddings
- `article` [Introduction to Word Embeddings](https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc)
- `article` [Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)
- `paper` word2vec: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Mikolov et al.
- `paper` GloVe: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) by Pennington et al. and [Stanford webiste for GloVe](https://nlp.stanford.edu/projects/glove/)
- `paper` fastText: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) by Joulin et al.
- `paper` [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf) by Cer et al., 2018

#### 6.3 Topic modeling
- `paper` LDA: [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  by Blei et al.
 - `paper` Anchored CorEx [Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge](https://arxiv.org/pdf/1611.10277.pdf) by Gallagher et al., 2017 and [github](https://github.com/gregversteeg/corex_topic)
 - `github` [Top2Vec](https://github.com/ddangelov/Top2Vec) and `paper` [Top2Vec: Distributed Representations of Topics](https://arxiv.org/pdf/2008.09470.pdf) by Dimo Angelov
 - `github` [BERTopic](https://github.com/MaartenGr/BERTopic) and `article` [Topic Modeling with BERT](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) by Maarten Grootendorst
 - `blog post` - `StitchFix` - [Introducing our Hybrid lda2vec Algorithm](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=) by Chris Moody

 #### 6.4 Transformers
 - `paper` transformers [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et al, 2017
 - `article` [The Illustrated GPT-2 (Visualizing Transformer Language Models](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar
- `article` [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) by Jay Alammar

### 7. Experimentation
#### 7.1 AB testing
- `book` [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264) by Kohavi, et al.
- `course` [Microsoft Experimentation Platform](https://exp-platform.com/2017abtestingtutorial/)
- [Evan Miller's A/B test tools](https://www.evanmiller.org/ab-testing/)
- `paper` [Three Key Checklists and Remedies for Trustworthy Analysis of Online Controlled Experiments at Scale](https://www.microsoft.com/en-us/research/uploads/prod/2020/06/2019-FabijanDmitrievOlssonBoschVermeerLewis_Three-Key-Checklists_ICSE_SEIP.pdf)
- `paper`[Top Challenges from the first Practical Online Controlled Experiments Summit](https://www.microsoft.com/en-us/research/uploads/prod/2020/07/2019-FirstPracticalOnlineControlledExperimentsSummit_SIGKDDExplorations.pdf)
- `paper` [Controlled experiments on the web: survey and practical guide](https://link.springer.com/article/10.1007/s10618-008-0114-1) Kohavi et al, 2009
- `article` [Guidelines for A/B Testing](https://hookedondata.org/guidelines-for-ab-testing/)
- `article` [A/B Testing: 29 Guidelines for Online Experiments (Plus a Checklist)](https://www.alexbirkett.com/ab-testing/#27)
- `course` [Udacity - A/B testing by Google](https://www.udacity.com/course/ab-testing--ud257)
- `article` [Evan Miller - Simple Sequential A/B testing](https://www.evanmiller.org/sequential-ab-testing.html)
- `article` [David Robinson - Understanding Bayesian A/B testing](http://varianceexplained.org/r/bayesian_ab_baseball/)
- `paper` [Overlapping Experiment Infrastructure:
More, Better, Faster Experimentation](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36500.pdf) by Tang et al, 2010

#### 7.2 Multi-Armed Bandits (MAB)
- `paper` [Best arm identification in multi-armed bandits with delayed feedback](https://arxiv.org/pdf/1803.10937.pdf)
- `paper` [Generalized Thompson Sampling for
Contextual Bandits](https://arxiv.org/pdf/1310.7163.pdf)
- `paper` [Analysis of Thompson Sampling for the Multi-armed Bandit Problem](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf)
- `paper` [A Contextual-Bandit Approach to
Personalized News Article Recommendation](https://arxiv.org/pdf/1003.0146.pdf)



### 8. Building web apps
- [Flask Mega-tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) by Miguel Ginberg

### 9. AWS
- `course` [Coursera - Getting Started with AWS Machine Learning](https://www.coursera.org/learn/aws-machine-learning)
- `course` [Coursera - AWS Cloud Technical Essentials](https://www.coursera.org/learn/aws-cloud-technical-essentials#syllabus)
- `course` [Coursera - Practical Data Science Specialization](https://www.coursera.org/specializations/practical-data-science#courses)
- [AWS Ramp up guide](https://d1.awsstatic.com/training-and-certification/ramp-up_guides/Ramp-Up_Guide_Machine_Learning.pdf)

### 10. Coding best practies
#### 10.1 Structuring projects
- `article` [Structuring Your Project: The Hitchhiker's Guide to Python](https://python-docs.readthedocs.io/en/latest/writing/structure.html)
- `github` [Cookiecutter data science](https://github.com/drivendata/cookiecutter-data-science)
- `blog post` [How to Set Up a Python Project For Automation and Collaboration](https://eugeneyan.com/writing/setting-up-python-project-for-automation-and-collaboration/) by Eugene Yan


#### 10.2 Code refactoring workflow
- `article` [The importance of structure, coding style, and refactoring in notebooks](http://blog.dominodatalab.com/the-importance-of-structure-coding-style-and-refactoring-in-notebooks)
- `tutorial` [Production Data Science](https://github.com/FilippoBovo/production-data-science)

#### 10.3 Unit testing
- `article` [Effective Python Testing With Pytest
](https://realpython.com/pytest-python-testing/) Real Python
- `article` [Becoming a Better Data Scientist: Testing with pytest](https://changhsinlee.com/pytest-intro/) by Chang Hsin Lee
- `article` [Unit Testing for Data Scientists](https://towardsdatascience.com/unit-testing-for-data-scientists-dc5e0cd397fb)

#### 10.4 Creating PyPI packages
- `tutorial` [PyPA Packaging Python projects tutorial](https://packaging.python.org/tutorials/packaging-projects/#python-requires)
- `e-book` [Python Packages e-book](https://py-pkgs.org/)
- `e-book` [The Joy of Packaging](https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html)
- [poetry](https://python-poetry.org/)
- `article` [How to Build Your First Python Package](https://towardsdatascience.com/how-to-build-your-first-python-package-6a00b02635c9)


### 11. Helpful other tools and packages
#### 11.1 Hyperopt
- `article` [Parameter Tuning with Hyperopt](https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce) by District Data Labs
- `article` [On Using Hyperopt: Advanced Machine Learning](https://blog.goodaudience.com/on-using-hyperopt-advanced-machine-learning-a2dde2ccece7) by Tanay Agrawal
- `article` [An Introductory Example of Bayesian Optimization in Python with Hyperopt](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0) by Will Koehrsen


### 12. Datasets
#### 12.1 General
- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)
- [Google Dataset search](https://datasetsearch.research.google.com/)
- [Registry of Open Data on AWS](https://registry.opendata.aws/)

#### 12.2 NLP
- [huggingface datasets](https://github.com/huggingface/datasets)
- [nlp-datasets github repo](https://github.com/niderhoff/nlp-datasets)

#### 12.3 Time series
- [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- [Time Series Classification Repository](http://www.timeseriesclassification.com/index.php)


### 13. Domain applications
#### 13.1 Rewewable Energy
- `paper` [Tackling Climate Change with Machine Learning](https://arxiv.org/pdf/1906.05433.pdf)
- [ClimateChage AI](https://www.climatechange.ai/)

#### 13.2 Healthcare
- `course` [MIT OCW Machine Learning for Healthcare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/) and [lecture videos](https://www.youtube.com/playlist?list=PLUl4u3cNGP60B0PQXVQyGNdCyCTDU1Q5j)

### 14. Additional topics
### 14.1 Ethics
- `github repo` [EthicalML Awesome Production ML](https://github.com/EthicalML/awesome-production-machine-learning)

### 14.2 Bias and explanability
- `package` [Fairlearn](https://github.com/fairlearn/fairlearn)

### 15. Other learning resource lists
- `article` [Data science learning resources](https://medium.com/data-science-at-microsoft/data-science-learning-resources-193ccf6fafb) by Microsoft Data Science team
- `blog` [End-to-End Machine Learning](https://e2eml.school/blog.html#000) by Brandon Rohrer (some good free resources, some paid)
- `github repo` [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)

### 16. Industry resources and trends
#### 16.1 Company tech blogs
- [AirBnb]()
- [Coursera]()
- [Stitch Fix]()
- `Square` [Product Analytics at Square](https://developer.squareup.com/blog/product-analytics-at-square/)

#### 16.2 Newsletters
- [Data Science Weekly](https://www.datascienceweekly.org/)
- [Papers With Code Newsletter](https://paperswithcode.com/newsletter)

#### 16.3 Podcasts
- [Data Skeptic](https://dataskeptic.com/)
- [Linear Digressions](http://lineardigressions.com/)
- [Talking Machines](http://www.thetalkingmachines.com/)

## Career resources
### Career advice
- `book` [Build a Career in Data Science](https://www.manning.com/books/build-a-career-in-data-science) by Emily Robinson and Jacqueline Nolis
- `article` [80000 hours: Data Science career review](https://80000hours.org/career-reviews/data-science/)
- `article` [Data science career advice to my younger self](https://towardsdatascience.com/data-science-career-advice-to-my-younger-self-4c37fac65184) by Schaun Wheeler
- `Quora` [As a data scientist, what career advice changed your life?](https://www.quora.com/As-a-data-scientist-what-career-advice-changed-your-life)
- `blog post` [A Framework for Career Decisions](https://www.conordewey.com/blog/career-decisions/) by Conor Dewey
- [ApplyingML - Mentor interviews](https://applyingml.com/mentors/) by Eugene Yan

### Defining data science
- `blog post` [Applied / Research Scientist, ML Engineer: What’s the Difference?](https://eugeneyan.com/writing/data-science-roles/) by Eugene Yan
- `reddit` [Difference between DS and MLE](https://www.reddit.com/r/datascience/comments/i48b5q/for_those_that_work_for_a_team_that_has_both_data/)

### Becoming a data scientist
- [Open-Source Data Science Masters](https://github.com/datasciencemasters/go)
- `article` [How to Build a Data Science Portfolio](https://towardsdatascience.com/how-to-build-a-data-science-portfolio-5f566517c79c)
- `github` [Awesome Data Science](https://github.com/academic/awesome-datascience)

### Generalist vs specialist 
- `article` [Unpopular Opinion - Data Scientists Should be More End-to-End](https://eugeneyan.com/writing/end-to-end-data-science/) by Eugene Yan
- `article` - `Stitch Fix` [Beware the data science pin factory: The power of the full-stack data science generalist and the perils of division of labor through function](https://multithreaded.stitchfix.com/blog/2019/03/11/FullStackDS-Generalists/) by Eric Colson

### IC vs Management and career progression
- `blog post` [Finding Answers to your Career Questions](https://hookedondata.org/career-resources/)
- `blog post` [Engieering Management: The Pendulum or the ladder](https://charity.wtf/2019/01/04/engineering-management-the-pendulum-or-the-ladder/) by Charity Majors
- `blog post` [The Engineer/Manager Pendulum](https://charity.wtf/2017/05/11/the-engineer-manager-pendulum/) by Charity Majors
- `blog post` [Senior engineer and then what?](http://www.juyang.co/senior-engineer-and-then-what/) by Ju Yang

### Team structure
- `article` [Models for integrating data science teams within organizations](https://medium.com/@djpardis/models-for-integrating-data-science-teams-within-organizations-7c5afa032ebd)
- `article` - `Coursera` [Analytics at Coursera: three years later](https://medium.com/@chuongdo/analytics-at-coursera-three-years-later-638498709ac8)
- `article` - `Coursera` [What is the most effective way to structure a data science team?](https://towardsdatascience.com/what-is-the-most-effective-way-to-structure-a-data-science-team-498041b88dae)
- `article` - `AirBnB` [At Airbnb, Data Science Belongs Everywhere](https://medium.com/airbnb-engineering/at-airbnb-data-science-belongs-everywhere-917250c6beba)
- `article` [Embedding Data Science In Cross-Functional Teams](https://medium.com/@Infinite_Monkey/embedding-data-science-in-cross-functional-teams-7bfce9283ad2)

### Data-driven culture
- `blog` [Building a data team at a mid-stage startup: a short story](https://erikbern.com/2021/07/07/the-data-team-a-short-story.html) by Erik Bernhardsson
- `blog` - `StitchFix` [Let Curiosity Drive: Fostering Innovation in Data Science](https://multithreaded.stitchfix.com/blog/2019/01/18/fostering-innovation-in-data-science/)

### Interviewing
- `book` [Introduction to Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/) by Chip Huyen

## Non-technical resources
### Agile & Project management
- `article` [How to manage Machine Learning and Data Science projects](https://towardsdatascience.com/how-to-manage-machine-learning-and-data-science-projects-eecacfc8a7f1)
- `article` [Data Science and Agile (What works, and what doesn't)](https://eugeneyan.com/writing/data-science-and-agile-what-works-and-what-doesnt/) and [Data Science and Agile (Frameworks for effectiveness)](https://eugeneyan.com/writing/data-science-and-agile-frameworks-for-effectiveness/)


### Product
- `article` [Jobs To Be Done Framework](https://medium.com/make-us-proud/jobs-to-be-done-framework-748c761797a8)
- `article series` - `Sequoia` [Data-Informed Product Building](https://medium.com/sequoia-capital/data-informed-product-building-1e509a5c4112)

### Business
- `article` [10 Reads for Data Scientists Getting Started with Business Models](https://www.conordewey.com/blog/10-reads-for-data-scientists-getting-started-with-business-models/) by Conor Dewey