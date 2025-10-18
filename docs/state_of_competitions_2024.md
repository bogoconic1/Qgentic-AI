# The State of  Machine Learning Competitions

### 2024 Edition

#### We summarise the state of the ML competitions landscape and analyse the hundreds of competitions that took place in 2024. Plus an overview of winning solutions and commentary on techniques used.

[A Jolt ML Publication](https://joltml.com/?ref=state24b)

## Highlights

- [**Over 400 ML competitions in 2024**, with more than $22m in total prize money, hosted on 20+ different platforms.](https://mlcontests.com/state-of-machine-learning-competitions-2024#ml-competitions-landscape)
- [**A resurgence of ‘grand challenge’ style competitions**, with over $1m in prize money each.](https://mlcontests.com/state-of-machine-learning-competitions-2024#grand-challenges)

- [**Python remains almost unchallenged** as the language of choice; PyTorch and gradient-boosted tree models were most common among winners.](https://mlcontests.com/state-of-machine-learning-competitions-2024#winning-toolkit)
- [**Quantisation proved key** in winning solutions for LLM-related competitions which tested skills like reasoning and information retrieval.](https://mlcontests.com/state-of-machine-learning-competitions-2024#nlp--sequence-data)

- [**AutoML packages are showing value in narrow applications**, but reports of Kaggle Grandmaster-level ‘agents’ are premature.](https://mlcontests.com/state-of-machine-learning-competitions-2024#automl)
- [**Performance on the ARC Prize competition was used as a yardstick for benchmarking frontier LLMs’ reasoning capabilities**.](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize)

- [**Over 400 ML competitions in 2024**, with more than $22m in total prize money, hosted on 20+ different platforms.](https://mlcontests.com/state-of-machine-learning-competitions-2024#ml-competitions-landscape)
- [**A resurgence of ‘grand challenge’ style competitions**, with over $1m in prize money each.](https://mlcontests.com/state-of-machine-learning-competitions-2024#grand-challenges)

- [**Quantisation proved key** in winning solutions for LLM-related competitions which tested skills like reasoning and information retrieval.](https://mlcontests.com/state-of-machine-learning-competitions-2024#nlp--sequence-data)
- [**AutoML packages are showing value in narrow applications**, but reports of Kaggle Grandmaster-level ‘agents’ are premature.](https://mlcontests.com/state-of-machine-learning-competitions-2024#automl)

- [**A resurgence of ‘grand challenge’ style competitions**, with over $1m in prize money each.](https://mlcontests.com/state-of-machine-learning-competitions-2024#grand-challenges)
- [**Python remains almost unchallenged** as the language of choice; PyTorch and gradient-boosted tree models were most common among winners.](https://mlcontests.com/state-of-machine-learning-competitions-2024#winning-toolkit)
- [**Quantisation proved key** in winning solutions for LLM-related competitions which tested skills like reasoning and information retrieval.](https://mlcontests.com/state-of-machine-learning-competitions-2024#nlp--sequence-data)
- [**AutoML packages are showing value in narrow applications**, but reports of Kaggle Grandmaster-level ‘agents’ are premature.](https://mlcontests.com/state-of-machine-learning-competitions-2024#automl)

## ML Competitions Landscape

We found over 400 ML competitions that took place in 2024, across more than 20 competition platforms. The total cash
prize pool across all relevant competitions we found was over $22m, up from $7.8m in 2023[1](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:1).

### Platforms

Many competition platforms saw significant user growth in 2024. Most platforms grew their user base by more than 25%
over the previous year, and some platforms more than doubled theirs. Kaggle remains the biggest platform both
by registered users (over 22 million) and total available prize money in 2024 (over $4m).

#### 2024 Platform Comparison

Once again, the open-source CodaLab platform hosted the most competitions in 2024 (113)[2](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:2), while
its successor, the newer Codabench platform[3](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:3), also hosted dozens of competitions and
quadrupled its user base in 2024.

DrivenData celebrated its 10th birthday in 2024, and the team posted some interesting reflections in a post titled
[10 takeaways from 10 years of data science for social good](https://drivendata.co/blog/10-years-of-data-science-for-social-good?ref=mlcontests).

#### Overview of Competition Platforms

| Platform | Launched | Users[4](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:4) | Competitions | Total prize money[5](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:5) |
| --- | --- | --- | --- | --- |
| [AIcrowd](https://aicrowd.com/challenges?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2017 | 245k+ | 11 | $113,000 |
| [Antigranular](https://antigranular.com/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2022 | 2.7k+ | 2 | $10,000 |
| [Bitgrit](https://bitgrit.net/competition?ref=mlcontests) | 2017 | 35k+[6](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:6) | 5 | $184,000 |
| [Codabench](https://codabench.org/?ref=mlcontests) | 2023 | 17k+ | 37 | $179,000 |
| [CodaLab](https://codalab.lisn.upsaclay.fr/?ref=mlcontests) | 2013 | 74k+ | 113 | $150,000 |
| [CrunchDAO](https://crunchdao.com/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2021 | 7k+ | 4[7](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:7) | $180,000 |
| [DrivenData](https://drivendata.org/competitions?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2014 | 125k+ | 9 | $650,000 |
| [EvalAI](https://eval.ai/?ref=mlcontests) | 2017 | 40k+ | 36 | $92,000 |
| [Grand Challenge](https://grand-challenge.org/challenges/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2010 | 109k+ | 37 | $56,000 |
| [Hugging Face](https://huggingface.co/competitions?ref=mlcontests) | 2023 |  | 11 | $138,000 |
| [Kaggle](https://www.kaggle.com/competitions?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2010 | 22m+ | 44 | $4,254,000 |
| [Signate](https://signate.jp/?ref=mlcontests) | 2014 | 80k+[8](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:8) | 14 | $99,000 |
| [Solafune](https://solafune.com/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2020 |  | 4 | $46,000 |
| [ThinkOnward](https://thinkonward.com/app/c/challenges?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2022 | 6.9k+ | 15 | $405,000 |
| [Tianchi](https://tianchi.aliyun.com/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2014 | 1.4m[9](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:9) | 45 | $659,000 |
| [Trustii](https://www.trustii.io/?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2020 | 2k+ | 4 | $57,000 |
| [Zindi](https://zindi.africa/competitions?ref=mlcontests) _Data compiled in collaboration with the platform team_ | 2018 | 75k+ | 21 | $113,000 |
| Other |  |  | 41 | $15,143,000 |

Note: the table above is shown with a reduced set of columns on mobile. For the full table, view this report on a larger screen.

Other platforms


For readability and relevance, the table above only includes platforms that hosted multiple competitions in 2024.
Decisions around what exactly constitutes a platform are somewhat subjective, and we remain open to changing this in future.
Competitions in the “Other” bucket in 2024 include:

- The DARPA AI Cyber Challenge
- The Vesuvius Challenge
- MIT Battlecode
- Numerous other competitions on special-purpose sites

This year’s platforms table includes two newer platforms which were not included in the 2023 table:

- [Antigranular](https://antigranular.com/?ref=mlcontests) hosts ML competitions that incorporate elements of
privacy-enhancing technologies, requiring competitors to balance high predictive accuracy with minimal use of privacy
budgets.
- Alongside its general ML competitions, [CrunchDAO](https://crunchdao.com/?ref=mlcontests) runs an ongoing competition
to predict returns for US equities, with the predictions used to manage a hedge fund portfolio.

### Grand Challenges

There has been a resurgence of interest in _grand challenges_ in machine learning: ambitious competitions centered
around difficult and impactful research problems which we don’t yet know how to solve. A few of these competitions have
been launched in recent years, with three of them funded mostly or entirely by individuals.

The AI Cyber Challenge, a two-year competition organised by DARPA in collaboration with several frontier AI labs,
explores the use of AI for improving cybersecurity. The semifinals took place at DEF CON 2024, where the top 7 teams won
$2m each, in addition to separate prizes for small businesses.

The [Vesuvius Challenge](https://mlcontests.com/state-of-machine-learning-competitions-2024#vesuvius-challenge) aims to uncover text from long-buried two-thousand-year-old papyrus scrolls using high-resolution
X-rays and machine learning. Building on the expertise of Brent Seales, who had already
been working on this project for several years, and with $250k in initial funding from Nat Friedman and Daniel Gross,
the project now has millions of dollars in funding from
[several individuals and organisations](https://scrollprize.org/#sponsors) and has awarded multiple progress prizes.
We covered 2023 results in some depth [last year](https://mlcontests.com/state-of-competitive-machine-learning-2023/#vesuvius-challenge), and provide an update below on the fifth
scroll scanned a few months ago, as well as the progress made in building automation tools.

The [AI Mathematical Olympiad](https://mlcontests.com/state-of-machine-learning-competitions-2024#ai-mathematical-olympiad), set up by the trading firm XTX Markets, has a $10m prize pot to spur the creation
of a publicly-shared, open source AI model capable of winning a gold medal in the International Mathematical Olympiad
(IMO). Hundreds of thousands of dollars have already been paid out, and the second progress prize competition is
currently ongoing. For insights into the winning strategy for the first progress prize and some changes for the second,
see below.

The [Abstraction and Reasoning Corpus (ARC) Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize), created and funded by François Chollet and
Mike Knoop, builds on previous ARC competitions, and aims to be “a barometer of how close or how far we are to
developing general AI”. This iteration has a $1m+ prize pool, and has drawn interest from ML researchers and startups.
There was significant progress on the state of the art in this competition in 2024, and we give a summary further down.

In the [recently-launched Konwinski Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#ongoing-and-future-competitions), the first team able to reach 90%+ on a new, dynamic, version of the SWE-Bench
code-generation benchmark wins $1m. Funded by Andy Konwinski, this competition evaluates LLMs on real-world software
issues collected from GitHub. The deadline for submissions is 12 March 2025.

### Corporates/Nonprofits

Aside from the grand challenges mentioned above, there were dozens of competitions in 2024 run by companies,
non-profits, or government organisations with the goal of solving a problem or hiring data scientists.

For example: the U.S. Bureau of Reclamation, who manage water resources, funded a
[series of competitions](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/?ref=mlcontests) with a total
of $500k in prize money, hosted on DrivenData, for making accurate water supply forecasts for 26 sites in the Western
U.S.

The Learning Agency Lab, a nonprofit focused on developing the science of learning-based tools, has run
[9 competitions](https://www.kaggle.com/organizations/learningagencylab/competitions?ref=mlcontests) on
Kaggle in the past 3 years (including 5 in 2024), with hundreds of thousands of dollars in total prize money.
Competition tasks included automating essay scoring, detecting AI-generated text, and detecting PII data.
Some of these competitions were funded with support from the Bill & Melinda Gates foundation, Schmidt Futures, and the
Chan Zuckerberg Initiative.

AI for Good, which describes itself as _“the United Nations’ leading platform on Artificial Intelligence for_
_sustainable development”_, has run multiple competitions on Zindi in the past few years. These competitions have focused
on problems like estimating air pollution from remote sensor data, and using satellite imagery for cropland mapping or
predicting soil parameters.

Both Meta and Amazon ran multitask competitions on AIcrowd in 2024, in association with the ACM KDD (knowledge
discovery and data mining) conference. Amazon’s competition focused on aspects of online shopping, while Meta’s
competition mainly dealt with web-based knowledge retrieval and summarisation.

### Academia

For some competitions affiliated with academic conferences, the goal is to enable researchers to directly compare
state-of-the-art methods on standardised datasets in a shared evaluation environment —
a helpful addition to papers where researchers replicate or cite each other’s results.

This year we identified hundreds of conference-affiliated competitions.
This includes competitions for conferences with official competition tracks — such as NeurIPS, MICCAI, and ICRA
— along with competitions affiliated with conference workshops, which tend to go through a less comprehensive
review process.

While some conference-affiliated competitions attract large numbers of submissions, many target a niche group of
researchers in a particular field. The barrier to entry can be higher, and winners are often rewarded with academic
kudos (in the form of a conference talk or paper) instead of, or in addition to, monetary prizes.

CodaLab hosted the largest number of conference competitions, including many of the computer vision competitions for
CVPR workshops.
Grand Challenge hosted the second-most, including dozens of biomedical imaging competitions for MICCAI.

EvalAI was third, with competitions across CVPR, ECCV, and many other conferences.

Several other platforms also hosted conference competitions — including Kaggle, which hosted competitions for
NeurIPS, CVPR, and other conferences, and AIcrowd, which hosted several tracks of competitions organised for ACM KDD.

#### NeurIPS Competitions

For NeurIPS 2024, competition organisers and participants from the previous year were
invited to submit papers to the
[datasets and benchmarks track](https://neurips.cc/virtual/2024/events/datasets-benchmarks-2024)
— a first for the conference.

Keep up with ML conferences


Get the latest technical updates from top machine learning conferences
including NeurIPS, ICML, and ICLR.

[Read Jolt's conference coverage →](https://joltml.com/conferences?ref=s24ctaa)

### Other Competitions

There were a few notable ML-adjacent competitions in 2024 which didn’t quite fit our [inclusion criteria](https://mlcontests.com/state-of-machine-learning-competitions-2024#methodology)
for this report:

- Google’s [Gemini API Developer Competition](https://ai.google.dev/competition?ref=mlcontests) had a $1m prize pool for
the best apps developed using the Gemini API.
- The open-ended [AI Agents Global Challenge](https://www.aiagentschallenge.com/?ref=mlcontests) also had a $1m prize pool,
split across investment and compute credits, for the best “AI Agents”.

### Prizes & Participation

The competition with the largest available prize pool was the AI Cyber Challenge, organised and funded by DARPA, with
$14m paid out to winners of the semi-finals that took place in August 2024.

Both the ARC Prize and the AI Mathematical Olympiad had available prize pools of over $1m. In each case some of the
prize pool was conditional on high absolute levels of performance being reached, and the actual amounts paid out were
$125k and $264k, respectively. These prize amounts are expected to roll over into the next editions of these
competitions.

#### Prize pool

Of the 18 competitions with prize pools of $100k or greater, 9 were hosted by Kaggle. Two were part of the Vesuvius
Challenge (the 2024 Grand Prize and First Automated Segmentation Prize). One was the AI Cyber Challenge.
The other six were on Bitgrit, Codabench, CrunchDAO, DrivenData, Humyn, and AI Singapore.

There were over 200 competitions with prize pools of at least $1k USD, of which more than half had prize pools of at
least $10k.

As always, there was a wide range of participation across competitions, with some niche conference competitions drawing
fewer than 10 teams (often researchers in a specific area), all the way to more mainstream competitions on the
larger platforms drawing over 5,000 teams.

#### Leaderboard entries

In general, competitions with more prize money tended to attract more participants. However, anecdotally, participants
often value other markers of success in competitions — such as academic kudos or competition platform progression
points — higher than monetary prizes. This is a longer-term outlook consistent with seeing competition success as
valuable professional experience. Some quant trading firms, for example, highly value success in ML competitions
among job applicants, and some even look for promising recruits on competition leaderboards[10](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:10).

## Winning Solution Trends

### Winning Toolkit

In line with previous years, Python was the almost-unanimous language of choice among competition winners.
Of the 79 winning solutions we found, 76 primarily used Python. There were three others:

- The winner of the [Polytope Permutation Puzzle](https://www.kaggle.com/competitions/santa-2023?ref=mlcontests), an
optimisation competition to solve Rubik’s-cube-style permutation puzzles in a minimal number of moves, implemented
[their solution](https://github.com/wata-orz/santa2023_permutation_puzzle?ref=mlcontests) in Rust. The second- and
third-place solutions also made extensive use of Rust or C++.
- The winner of [March Machine Learning Mania 2024](https://kaggle.com/competitions/march-machine-learning-mania-2024?ref=mlcontests),
a college basketball game prediction competition, was a high school science and stats teacher who used R to implement a
monte carlo simulation based on a combination of third-party team ratings and personal intuitions about the teams.
Winners of previous editions have also used R, as did the second-place solution this year.
- The [winner](https://github.com/igorkf/MLCAS2024) of the
[MLCAS Corn Yield Prediction](https://2024.mlcas.site/?ref=mlcontests#competition) competition, Igor Kuivjogi Fernandes,
used Python to preprocess the provided satellite data, but built their model (a linear mixed model) in R using the
`lme4` package. They told us that they used R because _“it is pretty strong for linear mixed models and I was_
_comfortable using it”_.

We analysed winning solutions’ code[11](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:11) where it was publicly available, and gathered information on packages used by some
teams who did not release their solutions’ code. The packages listed below represent the key third-party Python
packages that make up winning competitors’ core toolkit. Packages which were not included in last year’s toolkit are
highlighted.

#### Python Packages

Core

- numpy arrays
- pandas dataframes
- polars faster dataframes 🆕
- scipy optimisation and other fundamentals
- matplotlib low-level plotting
- seaborn higher-level plotting

NLP

- transformers tools for pre-trained models
- peft parameter-efficient fine-tuning
- trl reinforcement learning for language 🆕
- langchain LLM tools 🆕
- sentence-transformers embedding models 🆕

Vision

- opencv-python core vision algorithms
- torchvision core vision algorithms
- Pillow core vision algorithms
- albumentations image augmentations
- timm pre-trained models
- scikit-image core vision algorithms
- segmentation-models-pytorch segmentation

Modeling

- scikit-learn models, transforms, metrics
- deep learning
- torch
- tensorflow
- einops Einstein notation for tensor ops 🆕
- pytorch-lightning layer on top of PyTorch
- accelerate distributed PyTorch 🆕
- gradient-boosted trees
- lightgbm
- catboost
- xgboost

Other

- tqdm progress bar
- joblib parallelisation
- optuna hyperparameter optimisation
- psutil system tools
- loguru logs
- datasets loading data 🆕
- wandb experiment tracking
- shapely planar geometry 🆕
- rasterio geospatial raster data 🆕
- numba jit compilation

#### New Additions: Highlights

Many of this year’s popular Python packages were also popular last year. A few interesting packages
were more prominent in winning solutions this year than previously.

[einops](https://einops.rocks/?ref=mlcontests) provides an interface for tensor operations on top of NumPy, PyTorch,
TensorFlow, Jax, and other libraries.
This allows statements like `y = x.view(x.shape[0], -1)` to be replaced with statements like
`y = rearrange(x, 'b c h w -> b (c h w)')`; with dimensions given names.

[TRL](https://huggingface.co/docs/trl/en/index?ref=mlcontests) provides tools for language model post-training using
reinforcement learning, including techniques like Proximal Policy Optimisation and Direct Preference Optimisation.

[Accelerate](https://huggingface.co/docs/accelerate/en/index?ref=mlcontests) makes it easy to take PyTorch code written
for a single device and adapt it to run on a distributed setup, across multiple devices, with minimal changes.

[Shapely](https://shapely.readthedocs.io/en/stable/?ref=mlcontests) is a package to deal with planar geometric
objects as vectors, and can perform operations like checking whether a point is contained in a polygon (as used, for
example, in Matthew Aeschbacher’s
[winning solution for DrivenData’s Water Supply Forecast Rodeo](https://github.com/drivendataorg/water-supply-forecast-rodeo/blob/7f152a028a764cdc06b18a14568fabde298ade3e/overall/1st%20place/training/features/drought_deviation.py#L63)).

[Rasterio](https://rasterio.readthedocs.io/en/stable/?ref=mlcontests)
provides tools for dealing with geospatial raster data such as satellite imagery and terrain models in the TIF format.

### Deep Learning

#### Deep Learning: PyTorch vs TensorFlow (vs JAX?)

PyTorch remained the deep learning library of choice among winners, though TensorFlow was more prevalent than in prior
years. Out of the 60 winning solutions we found using deep learning, 53 used PyTorch and 7 used TensorFlow.

Almost all solutions that used TensorFlow did so through the higher-level Keras API. We found 3 uses of
Pytorch Lightning, and 1 of fastai. One solution used Keras directly, without TensorFlow.

The
[winners of the Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
competition primarily used PyTorch, however their solution used a small amount of JAX code (ported from an
[open-source implementation of the Superlets algorithm](https://github.com/irhum/superlets)) for transforming the time
series data. This was the only usage of JAX we found throughout 2024’s winning solutions.

### Computer Vision

Of the solutions using deep-learning-based computer vision architectures, 12 used convolutional neural
nets (CNNs), 5 used Transformers, and 3 used a combination of both.

#### Computer Vision Architectures

Within computer vision architectures, U-Net, ConvNeXt, and EfficientNet were the most common model families.

#### Model Families

In some ways, the framing of a competition problem limits the set of suitable architectures — with certain
architectures being designed for per-pixel segmentation or object detection, whereas others are tailored for regression
or classification of whole images.

Some problem specifications leave room for participants to define their own problem framing. For example, in Zindi’s
[Arm UNICEF Disaster Vulnerability Challenge](https://zindi.africa/competitions/arm-unicef-disaster-vulnerability-challenge/discussions/21520?ref=mlcontests),
participants needed to count the number of houses in each image with roofs made from certain types of roof material
(thatch/tin/other). The winning team combined two separate approaches: one framing this as an object detection problem,
where their models were trained to draw bounding boxes around a certain type of roof, which would then be counted.
In the second approach, they trained regression models to directly predict the number of houses of a
certain type in an image, without any intermediate object detection or segmentation.

### Compute and Hardware

As in
[previous](https://mlcontests.com/state-of-competitive-machine-learning-2023/?ref=state24#compute-and-hardware) [years](https://mlcontests.com/state-of-competitive-machine-learning-2022/?ref=state24#compute-and-hardware),
a majority of competition winners (over 80%[12](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:12)) used NVIDIA GPUs to train their models.
One used a Google TPU, through Google Colab, and the remainder used CPU resources only.

We did not find any instances of competition winners using accelerators other than CPUs, TPUs, or NVIDIA GPUs.
Notably, once again, we found no mention of AMD GPUs in winning solutions. This is line with trends in research
papers[13](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:13), which also rarely mention the use of AMD GPUs for training.
A December 2024 post by SemiAnalysis, based on extensive testing of AMD’s leading MI300X[14](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:14),
suggested that software considerations might be the main driver behind the lack of uptake of AMD’s GPUs for machine
learning, despite their cost advantage.

#### Hardware Used by Winners

In previous years, the NVIDIA A100 GPU was the most popular among winners by only a small margin. This year the A100
was more than twice as popular among winners as the next-most-popular GPU. Increased A100 availability (with the NVIDIA
H100 replacing it as the current top GPU for frontier model training) may have contributed to this leap in popularity.

#### Accelerator Models

Other than A100s, popular configurations include 1xP100 and 2xT4, which are available in Kaggle Notebooks. The most popular
consumer-grade cards were the RTX 4090 and RTX 3090.

We found two competition winners using 8xH100 nodes for training: _Numina_, who won the
[AI Mathematical Olympiad - Progress Prize 1](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/519303?ref=mlcontests),
and _The ARChitects_, winners of
the [ARC Prize 2024](https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf). This configuration costs
around $24/h using on-demand cloud compute[15](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:15), and is the most expensive configuration we found (on a per-hour
basis).

The winner of Kaggle’s [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106?ref=mlcontests)
competition, who goes by _c-number_, noted that they started with a local RTX 4090 GPU, but scaled up to renting 8x RTX
4090 when they realised they were running short on time.
They said that _“the investment in computational resources (approximately $500 in server_
_costs) was well justified by the final results”_.

Others also mentioned paying for cloud VMs or notebooks to train their solutions.
The winner of the [LEAP - Atmospheric Physics using AI (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/523063?ref=mlcontests)
competition, _greySnow_, mentioned that their experiments cost _“at least 200 bucks in Colab compute units and probably_
_less than 300 bucks in total”_. One other winner told us they spent over $100 renting an A100 to train their solution.

Despite some teams spending money on cloud compute or having access to clusters or powerful personal computers, there
were also winners who trained their solutions entirely for free through Kaggle or Colab notebooks.

#### Cloud Compute

Cloud compute services mentioned by winners included AWS, Jarvislabs.ai, Lambda, Runpod, TensorDock, and Vast.ai, with
one mention each.

Of the 10 competition winners we found using cloud notebooks for training, 7 used Kaggle Notebooks
(including 5 for non-Kaggle competitions) and three used Google Colab (one on the Pro tier, the other two on unknown
tiers).

As in previous years, there was significant variation in the amount of training data made available for competitions.
Some competitions provided only a few kilobytes of training data ( [AIMO Progress Prize 1](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-1)
provided 10 training examples).
On the other end of the scale, the
[DigiLut Challenge](https://www.trustii.io/post/join-the-digilut-challenge-advancing-lung-transplant-rejection-detection-1?ref=mlcontests)
came with multiple terabytes of lung biopsy data.

#### Dataset Size

The largest training datasets were usually for competitions with computer vision elements, or simulations of physical
systems. Competitions focused on reasoning/mathematics or NLP tended to have smaller training sets.

#### Training Time

Given the prevalence of ensembles and relatively large models, some solutions took multiple days to train. For
example, the winners of the
[Kelp Forest Segmentation](https://github.com/drivendataorg/kelp-wanted/blob/main/1st-place/)
challenge used 12 models which were each trained for 3-6 hours, and the winner of the
[Youth Mental Health Narratives](https://github.com/drivendataorg/youth-mental-health/tree/main/automated-abstraction/1st-place)
competition took ten days to train their final models.

It’s also possible to win a competition with minimal to no training. The winners of the
[SNOMED Entity Linking Challenge](https://github.com/drivendataorg/snomed-ct-entity-linking/blob/main/1st%20Place/reports/DrivenData-Competition-Winner-Documentation.pdf)
used a dictionary-based solution that took 6 minutes to train on CPU only. The winners of the [ICML 2024 Automated\\
Optimization Proble-Solving with Code](https://github.com/appliedai-spindox/icml24-automathres-t3) competition called
OpenAI’s GPT-4-turbo model at test-time, and their final solution did not include any training at all (they did
experiment with fine-tuning GPT-3.5-turbo, but did not use this in their final solution).

### Team Demographics

Most competitions allow individuals to team up and develop solutions together. Some platforms have a mechanism for
teams to advertise their openness to new members, and allow team mergers until shortly before the competition deadline.

Despite this, more than half of the winners we found in 2024 were individual competitors. Teams of more than 5
(a common upper limit for team size) were rare. While additional teammates can be helpful, having a team means
sharing any potential prize money, and some platforms explicitly incentivise individual entries as part of their
progression system.

#### Winning Team Sizes

Over half of the winning teams we found were categorised as ‘first-time winners’, as they did not have any members who
had already won a competition on the same platform[16](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:16).

#### Repeat Winners

As always, there were some very prolific teams and individuals.
For example, [Team Epoch](https://teamepoch.ai/?ref=mlcontests) from TU Delft won one competition on Zindi and one on DrivenData.
Kaggle Grandmaster [hyd](https://www.kaggle.com/hydantess?ref=mlcontests) had two solo wins on Kaggle.
Another Kaggle Grandmaster, [Ivan Panshin](https://www.kaggle.com/ivanpan), won a competition on Zindi as part of a team, and had a solo win on
Solafune.

## Winning Solution Specifics

This year, we’re spotlighting winning solutions to NLP competitions, and competitions around mathematics and reasoning.
See our [2023 report](https://mlcontests.com/state-of-competitive-machine-learning-2023?ref=state24) for more detail on time-series
competitions.

### NLP & Sequence Data

With the recent focus on generative modelling and the availability of increasingly powerful foundation models for
sequence data, many problems are being framed as sequence prediction problems.
Here we discuss traditional natural language processing (NLP) or sequence processing problems including text
extraction, text generation, sequence regression, sequence classification, and speech recognition.
We cover competitions focused on mathematics and reasoning tasks [below](https://mlcontests.com/state-of-machine-learning-competitions-2024#mathematics-and-reasoning).

The current focus in language model research is on autoregressive decoder models, which generate tokens one step
at a time. This is in contrast to encoder models, which take in strings of tokens and map them to a representation.

While encoder models have a history of success in NLP competitions,
[last year](https://mlcontests.com/state-of-competitive-machine-learning-2023/#nlp) we noted that decoder models were also starting to be used
successfully, both directly (e.g. by adding a classification head to a pre-trained model) or
indirectly (e.g. to generate synthetic data).

#### Decoder Models

This trend continued in 2024, and several competitions seemed designed specifically with these powerful new
decoder LLMs in mind.

The most commonly-used decoder models among competition winners in 2024 were variants of Llama, Mistral, Gemma, Qwen,
and DeepSeek models. Several competition winners used only decoder models.

The winners of AIcrowd’s
[KDD Cup 2024](https://discourse.aicrowd.com/t/winner-s-solution-overview-kdd-cup-2024-team-nvidia/16648?ref=mlcontests)
fine-tuned multiple Qwen2-72B models, making use of LoRA and 8 A100 GPUs at train-time and
4-bit quantisation and batch inference at test-time to enable such a large model to be used.

In the [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343?ref=mlcontests)
competition, the winner used an ensemble of Mistral-7B and Gemma-7B models. They, alongside other top-scoring teams,
used an adversarial attack on the competition metric that exploited a peculiarity in the vector embedding model, where
adding a certain token — for the string `lucrarea` — to the end of their submissions improved their scores
significantly.

In the [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106?ref=mlcontests)
competition, competitors had to build both question and answer agents, each of which would be paired off with another
competitor’s agent. These pairs of agents then needed to cooperate to guess the correct secret word using as few yes/no
questions as possible.
The overall winner, Kaggle user _c-number_, used agents with multiple strategies.
For their question-asking agent, they pre-populated a question table using questions generated by GPT-4o mini, and
probability distributions over answers calculated by sampling from Llama-3-8B-Instruct, Phi-3-small-8k-instruct, and
Gemma-7b-it. Their answering agent used Llama-3-8B-Instruct and DeepSeek-Math. No fine-tuning was done on any of
their models.
They were one of many teams who also implemented a simpler strategy purely based on alphabetical
bisection using regular expressions, which was more reliable but depended on the other (randomly assigned) agent in the
pair to also have implemented this strategy, and so potentially wasted one question on an initial ‘handshake’.

In
[LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629?ref=mlcontests), the winner ( _sayoulala_)
trained a Gemma2-9b model alongside two relatively large models (Llama3-70B and Qwen2-72B), using LoRA/QLoRA, after
which they used distillation to improve the smaller Gemma model (with the logits distribution from the larger models).
The smaller Gemma2-9b model was then quantised to 8-bit, and only that model was used for inference. They note that
“distillation is a very promising approach, especially in the current Kaggle competitions, where inference constraints
are a limiting factor”.

#### Encoder Models

Alongside decoder models’ success, encoder-only models still had a place in winning solutions. For the past
[few](https://mlcontests.com/state-of-competitive-machine-learning-2023/#nlp) [years](https://mlcontests.com/state-of-competitive-machine-learning-2022/#nlp-1),
the DeBERTa series of models have been the encoder-only model of choice among competition
winners, and this was still the case in 2024.

While some competition winners only used encoder models — such as the winner of the
[Automated Essay Scoring](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791?ref=mlcontests)
competition — it was more common to see encoder models be combined with decoder models.
One common way to combine them was to generate synthetic data using decoder models, and then use that to fine-tune the
encoder models.

In Kaggle’s
[PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374?ref=mlcontests)
competition, an ensemble of various DeBERTa models was trained on synthetic data generated by decoder (mainly Mistral
and Gemma) models. Similarly, the winner of AIcrowd’s
[Identifying Relevant Commonsense Facts](https://discourse.aicrowd.com/t/winner-s-solution-overview-commonsense-persona-grounded-challenge-2024/16646?ref=mlcontests)
competition fine-tuned a DeBERTa model on synthetic data generated using GPT-3.5-Turbo.

Another approach was to use an encoder for retrieval, paired with a decoder for generation, in a RAG
(retrieval-augmented generation) pattern. RAG can search over a corpus of documents to retrieve relevant text, which
is then added to the context window during generation.
The winner of Zindi’s [Specializing Large Language Models for Telecom Networks](https://arxiv.org/abs/2408.10808)
competition used ColBERT for retrieval, alongside Falcon-7.5B and Phi-2 for generation[17](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:17).

And lastly, it’s also possible to directly ensemble encoder and decoder models in a solution, as was done by the winner
of the
[Detecting AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121?ref=mlcontests)
competition, who used a combination of Mistral-7B, DeBERTa, LLama-7B, and Llama-1.1B, along with synthetic data.

Meta’s NLLB-200 encoder-decoder model was used by winners of two translation competitions on Zindi:
the [MLOps Machine Translation Challenge](https://zindi.africa/competitions/melio-mlops-competition?ref=mlcontests),
and [Tuning Meta LLMs for African Language Machine Translation](https://zindi.africa/competitions/tuning-meta-llms-for-african-language-machine-translation?ref=mlcontests).[18](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:18)

With the availability of recently-released encoder models like [ModernBERT](https://arxiv.org/abs/2412.13663),
it will be interesting to see if the popularity of DeBERTa endures into 2025.
The authors of the ModernBERT paper claim that it brings _“modern model optimizations to encoder-only models”_, and that
_“ModernBERT-base is the first encoder to beat DeBERTaV3-base since its release in 2021”_, while also being faster and
more memory-efficient.

#### Other Approaches

Not all NLP competitions were won using deep learning.
In the [SNOMED CT entity linking challenge](https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners),
competitors needed to tag clinical notes with relevant concepts. The winning team used a dictionary-based approach, and
narrowly beat the second-placed team which used an ensemble of BERT models, as well as comfortably beating other
teams which used decoder-only LLMs like Mistral-7B. The winning solution took under seven minutes to train, on CPU
only.

> Initially we aimed to develop an LLM-based solution, but as a baseline for comparison we also tried a simple dictionary-based approach.
> We proceeded in parallel, but fairly early on it seemed that in the context of this challenge, with the available annotated data, the dictionary approach was more promising.

— Team KIRIs, winners of the [SNOMED CT Entity Linking Challenge](https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners)

#### Quantisation and Model Size

There are two main axes of resource constraints on competition entries. Firstly, the cost of renting,
or acquiring and running, compute hardware for developing and training. Secondly, the compute constraints imposed by the
competition platform’s evaluation environment at inference time, often specified as a certain number of hours on a
specific hardware configuration (e.g. 1 NVIDIA P100 GPU with 16GB of VRAM for up to 12 hours).

In some cases, extra training compute can be expended to reduce the amount of inference compute required, by using
methods like model distillation and pre-calculating potential answers at train time.

Other techniques can be applied to reduce both train-time and inference-time compute. Two such techniques, which we
briefly mentioned [last year](https://mlcontests.com/state-of-competitive-machine-learning-2023/?ref=state24#adapters-and-quantisation),
are model weight quantisation and low-rank adapters.

We found instances of 4-bit, 5-bit, and 8-bit quantised models used in winning solutions.
Quantisation was usually used only for inference, but some teams quantised before fine-tuning.
This was particularly important for the winners of the [ARC Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize), who performed inference-time fine-tuning
(also referred to as _test-time training_ or _TTT_) within the confines of Kaggle’s evaluation environment.

Quantisation: the what and the why


Given that CPU-only inference is generally slow, it’s important for LLMs to fit within GPU memory at inference time to
make the most of the resources available.
There are two main dials that can be used to adjust the amount of memory a model uses: the number of parameters,
and the amount of memory taken up for each parameter.

Most models we saw used in winning solutions were in the 7-9 billion parameter range, though there were also some bigger
(e.g. Qwen2-72b) and smaller (e.g. DeBERTa, with up to 300 million parameters) models.

Parameters are usually stored as floating-point numbers. A few years ago, the default was to train models using
32-bit (4 byte) floats. With the more advanced datatypes supported by modern GPUs, as well as software improvements,
it’s now becoming possible to train using lower precision data types — 16 bit, and even 8 bit in some
cases[19](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:19).

Inference can be done at even lower precision, and often floating-point weights are _quantised_ into 8-bit or 4-bit
integer formats for inference (an 8-bit format can store 256 different values). Given rigid inference compute
constraints, quantisation trades off a (hopefully small) loss in model quality for an increase in the possible number of
model parameters, or an increase in generation speed and number of possible generations.

The winners of the [AIMO Progress Prize 1](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-1), team Numina, noted that the T4 and P100 GPUs
provided within
the standard Kaggle inference environment do not support _“modern data types”_ like bfloat16, and so they reduced the
precision of their model parameters to 8 bits using AutoGPTQ (noting also that _“casting \[from bfloat16\] to float16_
_leads to a degradation in model performance”_)[20](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:20).
They mentioned that quantisation _“led to a small drop in accuracy, but was mostly mitigated by being able to generate_
_many candidates during inference”_ — since quantisation makes inference significantly faster, as well as
reducing memory use. For the [second AIMO progress prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-2), Kaggle is allowing
participants to use more modern L4 GPUs, which do support bfloat16.

Another common technique to reduce memory requirements when fine-tuning is to freeze model weights and add a small
amount of new trainable parameters, through low-rank adapters (LoRA).
LoRA and related techniques were common among winning solutions that involved fine-tuning.
The winners of the [ARC Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize) noted that the Mistral-NeMo-Minitron-8B-Base model performed best
in their experiments. Because of its relatively high parameter count and the limited available GPU memory, they used
both LoRA and 4-bit quantisation when fine-tuning.

Not all winners chose to use LoRA. The winners of the [AIMO Progress Prize 1](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-1) stated
that they did not use LoRA or similar techniques because they _“were not confident they could match_
_the performance of full fine-tuning without significant experimentation”_. The team had access to significant
compute resources and used an 8x H100 node for training, which had enough GPU memory to do full fine-tuning.

### Mathematics and Reasoning

Two of 2024’s $1m+ prize pools were for competitions focused on mathematics and reasoning: the AI Mathematical Olympiad
(AIMO), and the Abstraction and Reasoning Corpus (ARC) Prize.

Team Numina won $131k for topping the leaderboard in the first AIMO Progress prize, which involved giving numerical
answers to natural-language maths questions.
Their solution involved using a powerful multi-GPU setup to fine-tune the _DeepSeekMath-Base-7B_ model, as well as
gathering hundreds of additional similar maths problems and solutions for validation.

The ARC Prize involves solving abstract grid-based puzzles.
The ARChitects’ winning solution for the 2024 edition of the ARC Prize was based around tokenising the grids into
1-dimensional sequences, and using an LLM to predict the outputs.

More detail on these two competitions and their winning solutions further down in the
[Notable Competitions](https://mlcontests.com/state-of-machine-learning-competitions-2024#notable-competitions) section.

In addition to these two competitions, there were several other mathematics and reasoning competitions. Winning
solutions to these also generally made use of LLMs, either by calling APIs for frontier models (when allowed), or by
fine-tuning pre-trained models with available weights.

In all three tracks of the ICML 2024 AI4Math Workshop’s _“Challenges on Automated Math Reasoning”_, participants
generally called LLM APIs rather than training or fine-tuning their own models, and two of the three tracks’ winning
solutions called the GPT-4 API. For more detail on these competitions, see our [ICML 2024 liveblog entry](https://joltml.com/icml-2024/ai-for-math-workshop-challenges/?ref=state24).

The [Global Artificial Intelligence Championships](https://www.agiodyssey.org/#/MathChampionship?Results=1) competition
had a $100,000 prize pool for the best performance on around 400 questions spanning high school,
college, and olympiad-level mathematics. Questions were provided in PDF and LaTeX formats. Answers were either true-false,
multiple-choice, or open-answer (real numbers/polynomial expressions/vectors/arrays). All answer formats could be graded
automatically.

> GAIC problem 149: What is the simplified value of the expression, 8x³ − 3xy + √p, if p = 121, x = −2, and y = 3/2?
>
> A) 84
>
> B) 73 + √11
>
> C) −28
>
> D) −44

32 teams submitted answers. The winning team did not disclose any details about their solution. The second-placed team
open-sourced [their solution](https://github.com/Jeff0741/GAIC-2024-Spring-EVS), which was based around GPT-4-Turbo.

### Time Series & Tabular Data

Unlike computer vision and NLP, where deep learning unlocked a step change in capabilities over previous approaches and
is now the obvious best option in most cases,
time-series and tabular data are two areas where deep learning methods are generally thought of as one of several
useful modelling approaches. To date, no single leading deep-learning-based architecture has emerged to dominate
time-series or tabular data problems.

On the whole, the techniques that won time-series and tabular data competitions in 2024 were not so different from those
that succeeded in previous years. Gradient-boosted decision trees were still extremely prevalent, and there was some
use of deep-learning-based methods.

For context on this section, see our in-depth looks at
[tabular data competitions in 2022](https://mlcontests.com/state-of-competitive-machine-learning-2022/#tabular-data)
and [timeseries forecasting competitions in 2023](https://mlcontests.com/state-of-competitive-machine-learning-2023/#timeseries-forecasting).

#### Gradient Boosted Trees

Gradient-boosted decision trees (GBDTs) largely continue to dominate competitions with tabular data or time-series
prediction which can be framed as tabular data.

We found 16 winning solutions using LightGBM, 13 using Catboost, and 8 using XGBoost, the three major GBDT
libraries. It was common, as in previous years, to see ensembles using multiple GBDT libraries, as their
varying implementations lead to different strengths and weaknesses in terms of modelling performance.

DrivenData’s
[Water Supply Forecast Rodeo](https://github.com/drivendataorg/water-supply-forecast-rodeo/blob/main/overall/1st%20place/),
the largest timeseries prediction competition of 2024 in terms of prize money, was won by Matthew Aeschbacher, who used
an ensemble of CatBoost and LightGBM models. The solution write-up describes initial experiments with XGBoost, which
they ended up replacing with CatBoost due to its superior handling of categorical features. LightGBM was also included
as part of the ensemble, as it _“trains extremely fast and offers predictive accuracy comparable to Catboost”_.

Aside from modelling performance, competitors might choose one library over another because of familiarity, or
because its implementation is better suited to a competition’s specific computational constraints.
This is why Kaggle user _hyd_, who used XGBoost to win the
[Enefit competition](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/472793),
opted to instead use CatBoost in their winning solution for the
[Optiver competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446). This
competition had a live evaluation period of three months following the submission deadline, during which submissions
were frozen but participants’ models could continue to train on newly collected market data.
They noted that _“During the private leaderboard phase \[for the Optiver competition\], we need to train online within_
_the Kaggle environment. XGBoost tends to run out of GPU memory, while CatBoost consumes less GPU and is also very fast.”_

#### Deep Learning

Another successful approach in tabular data competitions, as seen in previous years, was to build an ensemble
including both gradient-boosted trees and neural nets.

This can be seen in the [Home Credit competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337),
where the winner (SeungYun Kim) ensembled a LightGBM model, a CatBoost model, and a Denselight model (stack of MLPs with residual connections).
They noted: _“At first, I planned to lightly test with Denselight and then build the final model with a larger model_
_(FT-Transformer, etc.) \[…\] Surprisingly, I haven’t created or found a model that beats Denselight performance.”_
Regarding GBDTs, they also noted that LightGBM _“was a bit lacking compared to Catboost, but was good for the ensemble.”_

Similarly, in the
[Game-Playing Strength of MCTS Variants competition](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549801),
the winner ensembled a LightGBM model, a CatBoost model, and a TabM (deep learning) model. They noted that the TabM
model was their strongest single model in cross-validation.

In competitions incorporating time-series elements, winners have successfully used deep learning architectures including
Transformers and RNNs as in the [Optiver competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446),
CNNs as in the [Harmful Brain Activity Classification competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560),
and ensembles of all of the above in the [Flame AI Challenge](https://www.kaggle.com/competitions/2024-flame-ai-challenge/discussion/541397).

The winner of the [Linking Writing Processes to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/466873?ref=mlcontests)
generated features from sequences of events, and then built an ensemble of GBDT and deep neural nets
(TabNet and LightAutoML) on their table of features.

The summary blog post describing successful solutions for DrivenData’s
[Water Supply Forecast Rodeo](https://drivendata.co/blog/water-supply-forecast-and-final-winners?ref=mlcontests)
contains an example of an approach that is a more natural fit for deep learning than for tree-based models.
They note that, while first place and almost all other leading teams used gradient-boosted trees, the second-placed team
used a multi-headed multilayer perceptron (neural net) that simultaneously predicted multiple targets.
Not all gradient-boosted tree libraries support multi-target regression in this way, whereas with libraries like
PyTorch it’s easy to implement multi-target regression using a custom loss function.

While deep learning has clearly proved its use for tabular data, one notable omission among winning approaches in 2024
is the use of any tabular or time-series pre-trained foundation models.
These models, such as
Moirai and Chronos for time-series data
and TabPFN for tabular data,
are pre-trained on large amounts of data and can be used on novel data with zero to minimal fine-tuning, and in theory
could be used in the same way as pre-trained vision models like ConvNeXt and language models like Llama.

#### Dataframes

Pandas has long been the dominant dataframe library with few (if any) serious alternatives, but that has been changing
in recent years.
We found 7 winning solutions using Polars, up from 3 in 2023 and none in 2022. All of them also made at least some use
of Pandas. For some, Polars was at the core of their feature engineering pipeline. For others, Polars was just used for
a few peripheral I/O tasks.

Polars is implemented in Rust, and one of the benefits over Pandas is improved speed and memory efficiency.
In line with this, Kaggle user _hyd_, who won the Optiver and Enefit competitions by themselves, stated they used
Polars over Pandas for performance reasons.

> Polars is significantly faster than Pandas. All my feature engineering experiments are now written using Polars.
> However, I still use Pandas for some exploratory data analysis (EDA),
> though it probably only accounts for about 20% of my time.

— hyd


Polars hit version 1.0 in July 2024, with the team behind it confirming that this version number signifies
production-readiness[21](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:21), and support for Polars in mainstream ML libraries has been steadily improving.

For examples of how Polars was used in 2024, see the winning solutions for the
[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446),
[Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/472793),
[Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337),
and
[Building Instinct: Where power meets predictions](https://github.com/thinkonward/challenges/tree/main/energy-analysis/building-instinct/Casiopea)
competitions.

### AutoML

#### AutoML Competitions

Automated Machine Learning (AutoML) is a collection of techniques that aim to replace the subjective
human-driven parts of machine learning with automated processes. This includes designing features, choosing a suitable
model or ensemble of models, cross-validation, and hyperparameter optimisation.

Kaggle’s [2024 AutoML Grand Prix](https://www.kaggle.com/automl-grand-prix?ref=mlcontests) involved 5 tabular
data competitions where participants had 24 hours to develop their AutoML-based solutions. A Formula-1-style scoring
system was applied to the participants of each monthly challenge, from 25 points for 1st place down to 1 point for 10th
place, and a $75,000 prize pool was distributed across the five teams with the most points overall.

[In the end](https://www.kaggle.com/discussions/general/533161?ref=mlcontests), the top three teams were separated by a
mere two points. The teams in first and second place were made up of the developers behind the
[LightAutoML](https://lightautoml.readthedocs.io/?ref=mlcontests) and

[AutoGluon](https://auto.gluon.ai/?ref=mlcontests) libraries, respectively.
Fourth and fifth place were affiliated with
[H2O Driverless AI](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/?ref=mlcontests), and third place was an
individual, Robert Hatch, competing solo without any AutoML library affiliation.

The competition didn’t require solutions to be fully automated, just “AutoML-based”.
The value of the AutoML components in this competition came from giving competitors the ability to build
good models in a short space of time (within the 24h competition window).
Perhaps surprisingly, it wasn’t the case that the winning teams used only their own AutoML libraries —
it appears that they generally mostly used their own libraries, but also
at times used other tools such as gradient-boosted tree libraries or even other AutoML libraries.

The LightAutoML team noted that, in the two AutoML Grand Prix stages which they won, they used only LightAutoML.
The developers of AutoGluon maintain a
[list](https://github.com/autogluon/autogluon/blob/master/AWESOME.md#competition-solutions-using-autogluon) of Kaggle
competition solutions which make use of their library, and pointed out that 9 of the top 10 teams in the AutoML Grand
Prix used AutoGluon in a solution for at least one of the five stages.

We found two instances of other competition winners using LightAutoML in their solutions. The winner of Kaggle’s
[Linking Writing Processes to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/466873?ref=mlcontests)
competition used three LightAutoML neural net models, and the winner of Kaggle’s
[Home Credit competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)
used LightAutoML’s “Dense Light” model, in both cases as part of an ensemble.

#### Grandmaster-Level Agents?

Given ever-more-competent LLMs, in recent years there has been increasing excitement in the ML community about
LLM-driven autonomous “agents”.
One milestone for such systems is to compete in ML competitions at a level equivalent to expert humans.
This task is easier than many other real-world interaction tasks, in that the core interface is constrained to a few
websites and specific types of interactions.
On the other hand, given the breadth of modelling tasks included and the experience required to
avoid common pitfalls like overfitting to the public leaderboard, it is much harder than related
challenges like pure tabular AutoML or neural architecture search, and is still beyond the abilities of current
systems.

A November 2024 paper titled _“Large Language Models Orchestrating Structured Reasoning_
_Achieve Kaggle Grandmaster Level”_
presented a system that autonomously generated submissions to over 60 Kaggle competitions across tabular,
NLP, and computer vision tasks given just the competition URL.
These competitions were not, however, representative of the conditions and difficulty level required to earn a Kaggle
Grandmaster title[22](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:22), and
the authors clarify at the end of the paper that “this does not imply a formal Grandmaster title on Kaggle”
[23](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:23).

Some successful Kagglers, including quadruple Kaggle Grandmaster Bojan Tunguz, have commented on the paper and pointed
out the gap between the evaluations in the paper and the requirements to become a Kaggle Grandmaster.

> I can decidedly say that the claims of the “Kaggle Grandmaster Level Agent” are total unqualified BS.
> Main reason being that _none_ of the tests that they had done were done on an _active_ Kaggle competition,
> and the vast majority of the datasets they used were toy synthetic datasets for the _playground_ competitions.

— Bojan Tunguz, Quadruple Kaggle Grandmaster, [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7262147342366134273/)

Similar caveats apply to results on OpenAI’s [MLE-bench](https://openai.com/index/mle-bench/?ref=mlcontests)
benchmark. As addressed directly in the MLE-bench paper, _“not all our chosen competitions are medal-granting, MLE-bench_
_uses slightly modified datasets and grading, and agents have the advantage of using more recent technology than the_
_participants in many cases.”_

There is a way to sidestep all of these issues: having an AutoML system compete in active
competitions, with the same constraints as human competitors. The authors of the
above-mentioned November 2024 paper have stated their intent to do this in future work.

### External and Synthetic Data

Some competitions restrict participants to training models only using the data provided, whereas others allow the use of
_external_ data, often limited only to publicly available data.

Finding the right data to train on can be valuable, such as in Solafune’s
[Finding Mining Sites](https://solafune.com/competitions/58406cd6-c3bb-4f7a-85c7-c5a1ad67ca03?menu=discussion&tab=&topicId=7f557b23-2108-4686-8e2e-c3401c14d4c5)
competition, where only 1,000 annotated images were provided, and the winner relied on additional external datasets
containing a million images.

Even when external data is allowed, its use is not always necessary to win. The winner of Zindi’s
[Agricultural Plastic Cover Mapping](https://zindi.africa/competitions/geoai-challenge-for-agricultural-plastic-cover-mapping-with-satellite-imagery?ref=mlcontests)
competition, Tevin Temu, told us that they _“used a LightGBM classifier and focused on generating new features from the_
_provided dataset_”, rather than use any external data.

As we saw in [previous years](https://mlcontests.com/state-of-competitive-machine-learning-2023/#synthetic-data),
some competition winners used generative models to create additional _synthetic_ training data.

For example, in DrivenData’s [spacecraft detection](https://drivendata.co/blog/posebowl-winners) competition,
the winners first trained their model on 300,000 synthetic images, for which they generated backgrounds using a
diffusion model, before fine-tuning the model on the provided training data.

The winners of Kaggle’s [AI Mathematical Olympiad](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-1) also used significant synthetic data,
using GPT-4 to generate “reasoning paths”, which they then filtered before training their model on them.
Similarly, the winners of the [ARC Prize 2024](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-2024-winner) used synthetic data to supplement the mere hundreds of
training examples provided.
Other winners of NLP competitions used similar approaches — see [NLP & Sequence Data](https://mlcontests.com/state-of-machine-learning-competitions-2024#nlp--sequence-data) for
more.

### Use of API-gated models

The most capable frontier models — such as Claude, Gemini, and OpenAI’s various models — tend to be
available only via an API. This allows model providers to charge for usage, and prevents users from running a
copy of the model on their own hardware.

These models have been used by competition winners to generate synthetic data, and occasionally were called directly in
solutions at inference time.

However, these models can’t be incorporated in most competition solutions. Many of the most interesting recent
competitions are _code competitions_, where participants submit code that is evaluated on the
competition platform’s servers, as opposed to running their code locally and submitting predictions.
Usually, the evaluation environment for these code competitions does not allow solutions to call out to external APIs.
These conditions allow competition organisers to ensure that participants develop their solutions without any leakage.

Some competitions take a hybrid approach. The [ARC Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize), for example, has a _private_ leaderboard which
requires code submission as well as a _semi-private_ leaderboard (“ARC-AGI-Pub”) for which participants can generate
predictions locally and make use of model APIs. The _semi-private_ leaderboard allows models like OpenAI’s
o1 and o3 to be benchmarked on ARC — though the results are not directly comparable to the private leaderboard
results.

It is possible that some future competitions will enable competition organisers to keep their test data private without
requiring participants to share their model code or weights.
A [recent pilot experiment](https://openmined.org/blog/secure-enclaves-for-ai-evaluation/?ref=mlcontests) run by
OpenMined, in partnership with the UK AI Safety Institute and Anthropic, demonstrated a proof of concept for this type
of mutually private evaluation, using NVIDIA H100’s secure enclave technology.

## Notable Competitions

### AI Mathematical Olympiad

In a year when Google DeepMind announced a system that reached silver-medal-level performance in the International
Mathematical Olympiad (IMO), there was great focus on automated mathematical reasoning using language models.

XTX Markets’ [AI Mathematical Olympiad](https://aimoprize.com/?ref=mlcontests) (AIMO) competition series, first
announced in November 2023, aims to spur the open development of a model capable of winning a gold medal in the IMO,
with $5m set
aside for progress prizes and $5m for the grand prize. In 2024, a total of $263,952 was paid out for the first AIMO
progress prize.

#### AIMO Progress Prize 1

The annual IMO competition on which the AIMO is based requires answers to include proofs.
The scope of the first AIMO progress prize was more limited: the problems were easier than IMO-level
problems, and only involved integer solutions between 0 and 999 (inclusive). The question difficulty is described as
_“similar to an intermediate-level high school math challenge”_.

Ten sample questions were provided as training data; the public leaderboard throughout the competition was calculated
based on 50 questions, and the final (private) leaderboard based on another 50 questions. All the questions are
described as novel, and _“created by an international team of problem solvers”_.

As an example, one question from the AIMO progress prize 1 training set states: _“There exists a unique increasing_
_geometric sequence of five 2-digit positive integers. What is their sum?”_

Answer and reasoning


Answer: 211.

One possible line of reasoning: _We can write the geometric sequence as m\_i = n \* x^(i-1), with n an integer, and i ranging_
_from 0 to 5. The fifth number of this sequence is m\_5 = n \* x^4. For the m\_i to be integers, x needs to be a rational_
_number, so can be written as p/q, with p and q integers. q^4 must divide n (for m\_5 to be an integer),_
_so we can write n as n = k \* q^4._
_Picking the smallest possible numbers here gets us k=1, and q=2. p/q needs to be greater than 1 for the sequence to be_
_increasing, so the smallest p we can pick is 3. That results in a sequence of \[16, 24, 36, 54, 81\] whose_
_sum is 211._

The baseline solution, using Google’s open-source Gemma-7B model, scored 3/50 on the public and private test sets.
The majority of the prize pool was set aside for any team able to achieve a score of over 47/50 on both test sets.
The best score achieved during the competition was 29/50, by team Numina, with an impressive lead over the rest of the
field.
Second-placed CMU\_MATH got a score of 22, and only ten other teams (out of over 1,000) got a score of 20 or higher.

#### Winning Solution

Team Numina’s solution can be summarised as:

- Gathering a large dataset of hundreds of thousands of relevant mathematics problems with corresponding solutions.
- Using GPT-4 to generate additional solutions to some of the gathered problems, with reasoning paths that incorporate
tool use (e.g. symbolic solvers), and filtering out any solutions where the final answer was incorrect.
- Fine-tuning _DeepSeekMath-Base-7B_ (a base language model trained for solving mathematics problems) on the above
datasets — first to generate chain-of-thought solutions and then to generate solutions that make use of external
tools within a Python environment. Fine-tuning was done using a node of 8xH100 GPUs to fine-tune all the weights of
their language model without having to use techniques like LoRA.
- At inference-time, generating 48 candidates for each problem, with up to 3 iterations of calling out to a Python
environment, before using majority voting to choose an answer. Quantising their model to 8-bit precision allowed for
efficient inference and more generations than would have been possible at full precision.
- Avoiding overfitting by evaluating on four validation sets with around 1,600 problems from AMC, AIME, and the MATH
data sets.

Two key papers that influenced their approach are
[ToRA](https://arxiv.org/abs/2309.17452) (Tool-integrated Reasoning Agent) and
[MuMath-Code](https://arxiv.org/abs/2405.07551) (multi-perspective data augmentation combined with code-nested
solutions).

They have published their fine-tuned model, [NuminaMath-7B-TIR](https://huggingface.co/AI-MO/NuminaMath-7B-TIR), as well
as their two training datasets: [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) and
[NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR).

> We \[used\] a pipeline leveraging GPT-4 to generate TORA-like reasoning paths,
> executing the code and producing results until the solution was complete.
> We filtered out solutions where the final answer did not match the reference
> and repeated this process three times to ensure accuracy and consistency.

— Team Numina [blog post](https://huggingface.co/blog/winning-aimo-progress-prize)

After winning the first progress prize, Project Numina were awarded a €3m grant by XTX Markets to support their
initiative to collect and publicly release a dataset of up to one million formal mathematical problems and proofs.
They are not eligible to enter the second progress prize competition.

#### AIMO Progress Prize 2

The
[second AIMO progress prize](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2?ref=mlcontests)
launched in October 2024, and will close on the 25th of March, 2025. This prize has an available prize pool of over $2m,
as well as having harder problems ( _“around the National Olympiad level”_). Answers are still purely numerical —
integers between 0 and 999, though for some questions this will require taking the ‘real’ answer modulo 1000.
The performance threshold for unlocking the $1m+ top prize is set at 47/50 correct solutions on both the public and
private test sets. At the time of publication, the highest score on the public leaderboard is 31/50.

In the first AIMO progress prize, teams were allowed to use _“AI models and tools that are open source and were released_
_prior to 23 February 2024”_. All of the top 4 teams based their solutions around variants of the DeepSeek-Math-7B model.
All 4 teams also acknowledged that their solution was based at least in part on the public notebook shared by the
[winner of the $10k early sharing prize](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/497136?ref=mlcontests),
which incentivised open solution sharing among competitors.

In line with Numina’s suggestions in their post-competition blog post to _“enable evaluation on modern GPUs”_ and
_"\[relax\] the pretrained model cutoff date"_, a few changes have been made for the second AIMO progress prize.
Firstly, there is a [whitelisting process](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/548129?ref=mlcontests)
for models that can be used in solutions, enabling teams to use more-recently-released models.
Secondly, evaluation can now be done on
[virtual machines with 4x L4 GPUs](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/540909?ref=mlcontests).
These have a total of 96GB of GPU memory, as compared to 32GB for the 2xT4 GPU or 16GB for the P100 GPU VMs that Kaggle
competitions usually allow. Being a more modern GPU, the L4 also supports the bfloat16 data type and other optimisations
that are useful for efficient inference.

These two modifications enable solutions which are closer to state-of-the-art research in mathematical reasoning with
LLMs.

The $20k
[early sharing prize](https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference?ref=mlcontests)
for the second AIMO progress prize was won by a notebook that achieved a score of 20/50 on the public leaderboard using
the QwQ-32B-preview model — a “reasoning” model which makes use of inference-time compute scaling —
with some prompt engineering and inference tweaks, but without any fine-tuning.

The new ongoing whitelisting process gives onlookers a potential clue as to which models might be used by top teams.
On Thursday 23rd January 2025, a few hours before DeepSeek R1 was whitelisted for this competition[24](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:24),
the highest score on the public leaderboard was 25, and there were 42 teams on the leaderboard with a score of 21 or
above.
By the following Monday morning, team _NemoSkills_’ score had jumped from 23 to 27 with four additional submissions,
leapfrogging them into first place, and there were 82 teams with a score of 21 or over.
As of the time of publication, _NemoSkills_ are still at the top of the leaderboard, with a score of 31.

Interested in how machine learning can be used for mathematics?


Recent progress in mathematical reasoning using language models and reinforcement learning
have lead to impressive results.

Discover more about formalisation and how systems like DeepMind’s AlphaProof and AlphaGeometry work.

[Read the deep dive →](https://joltml.com/ml-mathematics/?ref=s24ctam)

### ARC Prize

One of the most-discussed competitions of 2024 was the ARC Prize, based on the ARC-AGI-1 data set. Described by
the dataset creator François Chollet as _“a barometer of how close or how far we are to developing general AI, and_
_second, a source of inspiration for AI researchers”_, the ARC Prize is made up of sequences of 2d-grid-based visual
puzzles, similar to the non-verbal reasoning puzzles used in intelligence tests.

In [last year’s report](https://mlcontests.com/state-of-competitive-machine-learning-2023/#arcathon), we gave an overview of the previous
iteration of the ARC challenge.

**Example 1: Input**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex1-q.svg)

**Example 2: Input**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex2-q.svg)

**Example 3: Input**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex3-q.svg)

**Test: Input**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-test.svg)

**Example 1: Output**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex1-a.svg)

**Example 2: Output**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex2-a.svg)

**Example 3: Output**![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-ex3-a.svg)

**Test: Output**

![](https://mlcontests.com/state-of-machine-learning-competitions-2024/images/arc-q.svg)

An example ARC task. Source:
[arcprize.org](https://arcprize.org/play?task=1caeab9d)

Even though the challenge had been around for almost five years in some form with the same private test set,
the added gravitas of the new $1m prize pool seemed to attract the attention of research labs who saw ARC Prize 2024
as an opportunity to evaluate and demonstrate the reasoning abilities of their systems — particularly at a time
when _reasoning_, in various forms, was seen as a weakness of language models.

> Many well-funded startups \[have\] shifted priorities to work on ARC-AGI
> — we’ve heard from seven such companies this year.
> Additionally, multiple large corporate labs have now spun up internal efforts to tackle ARC-AGI.

— [ARC Prize 2024: Technical Report](https://arxiv.org/abs/2412.04604)

There was significant progress on the ARC-AGI private leaderboard, with the best single-solution score going from
30% in 2023 to 55.5% in 2024. Since the 85% human-equivalent performance threshold was not reached, only $125k of the
total $1.1m available prize pool was paid out in 2024.

#### ARC 2024 Winner

The top-performing team, _the ARChitects_, scored 53.5% on the private test set — beating second-place’s 40% score
and a significant jump in performance from the 30% reached by the two joint winners of 2023’s ARCathon.
There was one team that got a higher score of 55.5%[25](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:25), but they were not eligible
for prizes as they chose not to open-source their solution.

![Flowchart showing the various stages of the ARChitects’ solution. It starts with a base model, which is fine-tuned with augmented public ARC dataset data.The resulting “preliminary model” is further fine-tuned on augmented private ARC dataset data, into a final model.This final model is used to generate multiple candidates for inference, and to score augmented versions of thecandidates. The best-scoring candidate is then submitted as their final prediction.](https://mlcontests.com/ARChitects_solution.png)Overview of the ARChitects’ solution. Source: [GitHub repo](https://github.com/da-fr/arc-prize-2024?tab=readme-ov-file).

The ARChitects’ winning solution has several interesting components:

1. Tokenisation: they tokenise the problem definitions into a 1-dimensional sequence, one token per cell, allowing them
to be consumed by a language model. They restrict the number of possible tokens to 64, with special tokens
including `newline` (to be able to represent a 2d grid) and others to delimit the beginning and end of input and output definitions.
2. Fine-tuning: they fine-tune a language model ( _Mistral-NeMo-Minitron-8B-Base_) on the provided training set and additional
data to predict output grids. At test time, they do additional fine-tuning on the private test data, and use LoRA,
4-bit quantisation, and checkpointing to enable this to happen within the limited compute budget of Kaggle’s evaluation
environment.
3. Candidate generation: they sample 8-16 output candidates from their fine-tuned model for each task, using depth-first search[26](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:26).
Depth-first search efficiently finds an output string that is assigned high likelihood by the model, whereas
greedy decoding could optimise for the likelihood of individual tokens at the expense of the string as a whole, and
stochastic sampling would be too computationally expensive.
4. Candidate selection: they select the 2 ‘best’ candidates for submission by choosing those for which the language model expresses the
highest confidence across a set of augmentations.

Their training data includes examples from the Re-ARC, Concept-ARC, and ARC-Heavy datasets, as well as augmentations
of these (rotation, reflection, shuffling the order of examples, and permuting colours). Augmentation is key throughout
the fine-tuning, candidate-generation, and candidate selection parts of their solution.

Notably, their selected augmentations are designed so that they preserve the task’s structure while significantly
changing the 1d tokenised representation that is fed into the model. They find that the model is better at learning
patterns when presented problems in certain orientations, and the augmentations allow them to exploit this
characteristic.

Augmentation is also used at test time. Each of their 8-16 candidate solutions corresponds to a prediction for one of 16
different augmentations of the task. Candidate selection is key: they show that on the public eval set, one particular
configuration of their algorithm contains the correct solution within the 16 generated candidates 80% of the time,
but a naive selection strategy would have the 2 selected candidates be correct only 60.5% of the time.

As for the language model used, they noted that modern architectures and larger models tended to perform better, and
several optimisations were made to allow them to use the largest model feasible within the evaluation parameters.

For more detail on the ARChitects’ solution, see [their write-up and code](https://github.com/da-fr/arc-prize-2024).

#### OpenAI o3 and DeepSeek R1-zero

There was also great progress on the ARC-AGI-Pub leaderboard, where progress on the “semi-private” evaluation set
— whose 100 task definitions and solutions were not released publicly, but task definitions were exposed to
participants who could use LLM APIs and internet access in their approaches — jumped from 43% in June 2024 to
53.6% in early December, before OpenAI’s o3 system achieved 75.7% in late December.

While the ARC-AGI private leaderboard limits inference compute and does not allow internet access during evaluation, the
only limitation on the public leaderboard is a $10,000 maximum inference cost limit.

OpenAI’s 75.7% score was achieved with an inference cost of $8,689 across the 500 tasks (100 semi-private and 400
public), or around $20 per task. OpenAI revealed a separate configuration of o3 that could achieve
87.5% on the semi-private evaluation set with “roughly 172x” as much inference compute — therefore not eligible
for the leaderboard, but nonetheless showing that there is room for performance improvement with additional compute.

The ARC Prize team also benchmarked DeepSeek’s R1 model on the semi-private evaluation set, and it achieved a score of
15.8% with an inference cost per task of $0.06.
Interestingly, R1-Zero — the version of DeepSeek’s R1 system which didn’t
go through the supervised fine-tuning stage requiring extensive human-labelled data — performed only slightly
worse, achieving a 14% score with an average cost of $0.11. It’s worth noting that the R1 systems were not adapted
specifically for the ARC task, whereas it is unclear how similar the o3 system evaluated on ARC is to the general o3
system[27](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:27).

#### Effective Approaches

The [ARC Prize 2024 Technical Report](https://arxiv.org/abs/2412.04604) gives a great overview of the variety of
approaches to solving ARC, on both the public and private leaderboards.

There were three major categories of successful approaches:

- Deep learning-guided program synthesis: using LLMs to generate Python programs that generate solutions.
Some enhancements to this included iterative debugging (providing the LLM with the output of the programs and letting
it propose edits), and specifying the programs in a domain-specific language (DSL). In contrast to a
general-purpose programming language like Python, the DSL would be designed to abstract away relevant concepts for ARC
like “objects”, “borders”, or “enclosing”, allowing program search over higher-level concepts.
- Test-time fine-tuning on models that directly predict outputs: updating the weights of the LLM at test time, by
fine-tuning either on the entire private test set or on the specific task at hand. Interestingly, this seems to allow
models to perform much better than just using the “in-context learning” approach of providing the test set examples in
the prompt. The ARChitects’ [#1 private leaderboard solution](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-2024-winner) discussed above used this approach.
- Combining the two approaches above: each of the two above approaches seemed to do particularly well at different
subsets of tasks, making ensembling them very effective. In the report, the program synthesis approach is generally
referred to as “inductive”, whereas directly predicting outputs is referred to as “transductive”.

#### Next Steps

A new dataset, ARC-AGI-2, is expected to launch alongside ARC Prize 2025. The organisers have stated that they are
committed to running the Grand Prize competition _“until a high-efficiency, open-source solution scoring 85% is created”_.

They have also learnt from the experience of running competitions based on ARC-AGI-1 over the last few years, and are
making ARC-AGI-2 harder for automated systems while still being easy for humans, stating that
_“early data points suggest that the upcoming ARC-AGI-2 benchmark will still pose a significant challenge to o3,_
_potentially reducing its score to under 30% even at high compute (while a smart human would still be able to score over_
_95% with no training). “_

### AI Cyber Challenge

DARPA — the US Defense Advanced Research Projects Agency — has a long history of running competitions with
large prizes, such as its 2004 $1m autonomous driving
_DARPA Grand Challenge_, of which some participants later went on to found self-driving car companies including Waymo
and Cruise.

In the recent DARPA AI Cyber Challenge, run in partnership with Anthropic, Google, Microsoft, OpenAI, and others,
competitors build tools to automatically find and fix vulnerabilities, using LLMs.
Challenge projects in the semi-finals were based on open source projects including Jenkins, the Linux kernel, Nginx,
SQLite3, and Apache Tika.

The submitted tools discovered 22 unique synthetic vulnerabilities, of which they patched 15. Notably, they also found
one real-world bug in SQLite3.

Andrew Carney, program manager for the competition, said of the semifinals that
_“we’ve seen that AI systems are capable of not only identifying but also_
_patching vulnerabilities to safeguard the code that underpins critical infrastructure”_ [28](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:28).

The final competition will take place at DEF CON 2025 in Las Vegas, where the winning team will receive
a $4m prize. For more on the AI Cyber Challenge, see [their website](https://aicyberchallenge.com/?ref=mlcontests).

### Vesuvius Challenge

As covered in detail in
[last year’s report](https://mlcontests.com/state-of-competitive-machine-learning-2023/?ref=state24#vesuvius-challenge),
the Vesuvius Challenge is an ongoing effort to extract text from papyrus scrolls that were carbonised when mount
Vesuvius erupted almost two-thousand years ago. The scrolls are scanned in minute detail using X-ray tomography, and the
resulting masses of 3D data are analysed to first _digitally unwrap_ each scroll into a flat sheet, then locate ink on
that sheet, and lastly identify the (Greek) letters and passages written in the ink.

In 2023, the team (and competition participants) managed to read over 5% of one of the scrolls, using a mixture of
machine learning/computer vision techniques and hundreds of hours of manual labour. One of the 2024 prizes was targeted
at repeating this feat, but with increased automation: bringing the human labour down to below 4 hours, while
maintaining at least 95% of the result. While this wasn’t fully achieved in 2024, good progress was made towards this
goal and partial prizes were paid out for two submissions.

The 2024 grand prize — to read over 90% of 4 of the scrolls — went unclaimed. Interestingly,
some scrolls appear to be much easier to read than others: while roughly 5% of the first scroll’s
text was recovered in the first year of the competition, even after two years no text has been recovered from the other
three scrolls that made up the initial dataset. In November 2024, a fifth scroll was scanned, ink was
discovered almost immediately, and some letters were visible with minimal effort[29](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:29).
Great news, as per the organisers:
_“Scroll 5’s greatest gift might be its potential ability to operate as a ‘Rosetta Stone’ for ink detection into other_
_scrolls.”_ [30](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:30)

Monthly progress prizes in the range of tens of thousands of dollars were paid out throughout 2024 for contributions to
various aspects of uncovering text within the scrolls, and in total around $1.5m in prize money has now been paid out
as part of the Vesuvius Challenge. The initial 2025 prizes include $200k for reading an entire scroll, and $60k for the
first team to uncover at least 10 letters within a small area of scrolls 2, 3, or 4.
The team behind the challenge are on a long-term mission to uncover huge amounts of writing from antiquity,
and are [hiring](https://scrollprize.org/jobs?ref=mlcontests) for several jobs including a platform engineer, and a
computer vision & geometry researcher.

## Looking Ahead

### Inference-time scaling

Some recent frontier models — notably OpenAI’s o1 and o3 models as well as DeepSeek’s R1 model —
have demonstrated the effectiveness of using larger amounts of inference-time compute to increase performance, also
referred to as _inference-time scaling_.

For code competitions, where inference happens on competition platforms’ servers, scaling up inference-time compute
becomes expensive, which may prove an interesting challenge for competition platforms.

On the one hand, there has long been an upward trend in compute usage for ML, and platforms like Kaggle have kept
up with this, by first adding K80 GPUs, then P100 GPUs, then 2x T4 GPUs. For both the Konwinski Prize and the second
AIMO progress prize, Kaggle is allowing competitors to use VMs with 4x L4 GPUs for evaluation. These have a total of
96GB of VRAM; 3x as much as the older VMs with 2x T4 GPUs.

On the other hand, the logarithmic inference-time compute scaling trends demonstrated by OpenAI’s o1 model seem to have
[shifted the ML research community’s focus](https://joltml.com/neurips-2024/highlights-inference-time-compute/?ref=mlcontests)
to finding effective ways to increase inference-time compute, and so the inference component of solutions’ compute
budget could end up scaling much faster than in previous years.

### Ongoing and future competitions

There are many interesting competitions in the works for 2025 and beyond, with some already live.
The second iteration of the [AI Mathematical Olympiad](https://mlcontests.com/state-of-machine-learning-competitions-2024#aimo-progress-prize-2) is currently running, some new
[Vesuvius Challenge](https://mlcontests.com/state-of-machine-learning-competitions-2024#vesuvius-challenge) prizes have been announced, and the
[ARC Prize](https://mlcontests.com/state-of-machine-learning-competitions-2024#arc-prize) is also expected to have a new competition launching soon.

The [Konwinski prize](https://www.kaggle.com/competitions/konwinski-prize?ref=mlcontests) builds on the SWE-Bench
benchmark, which measures models’ abilities to correctly submit code patches that close GitHub issues.

> I’m giving $1M to the first team that exceeds 90% on a new version of the SWE-bench benchmark containing GitHub issues
> we collect after we freeze submissions.

— [Andy Konwinski](https://www.kaggle.com/competitions/konwinski-prize?ref=mlcontests?ref=mlcontests)

The submission deadline is 12 March 2025. The evaluation set will be made up of new GitHub issues resolved in the three
months following the deadline.

## About This Report

### About ML Contests

For over five years now, [ML Contests](https://mlcontests.com/) has provided a competition directory and shared insights
on trends in machine learning competitions. To receive occasional updates with content like this report,
subscribe to our mailing list or RSS feed.

Powered by [EmailOctopus](https://emailoctopus.com/?utm_source=powered_by_form&utm_medium=user_referral)

If you enjoyed reading this report, you might like the previous editions:

[![/images/report-header-2023.png](https://mlcontests.com/images/report-header-2023_hu13256396777155453348.jpg)\\
\\
**State of Competitive ML 2023**\\
\\
\\
Last year's report, where we summarised results from 300+ competitions in 2023.](https://mlcontests.com/state-of-competitive-machine-learning-2023/) [![/images/Report Header.png](https://mlcontests.com/images/Report%20Header_hu13145735043479384165.jpg)\\
\\
**State of Competitive ML 2022**\\
\\
\\
Last year's report, where we summarised results from 200+ competitions in 2022.](https://mlcontests.com/state-of-competitive-machine-learning-2022/)

### About Jolt

ML Contests is part of [Jolt](https://joltml.com/?ref=state24joltabt), an online magazine for ML researchers and
practitioners.

Jolt publishes [long-read technical articles](http://joltml.com/ml-mathematics?ref=state24joltabt), as well as a blog with updates from
[major machine learning conferences](http://joltml.com/conferences?ref=state24joltconf).

### Acknowledgements

Thank you to Peter Carlens, Alex Cross, James Mundy, and Mia Van Petegem, for helpful feedback on drafts of this report.
Thank you to Fritz Cremer, for sharing insights on Kaggle competitions, and to Eniola Olaleye for help with data
gathering.

Thank you to the teams at AIcrowd, Antigranular, CrunchDAO, DrivenData, Grand Challenge, Kaggle, Solafune, ThinkOnward,
Tianchi, Trustii, and Zindi, for providing data on their 2024 competitions and winning solutions.

Thank you to the competition winners who took the time to answer our questions by email or questionnaire:
Igor Kuivjogi Fernandes, Jan Disselhoff, Lewis Hammond, Ivan Panshin, hyd, Gaius Doucet,
Harshit Sheoran, Stefan Strydom, Guberan Manivannan, Kranthi, Saket Kunwar, Tevin Temu, Alex Gichamba,
Tewodros K Idris, Brian Ebiyau, Eric Nyberg, Teruko Mitamura, Victor Tolulope Olufemi,
Aymen Sadraoui, and Charles Yusuf.

Lastly, thank you to the maintainers of the open source projects we made use of in conducting our research and
producing this page: [Hugo](https://gohugo.io/?ref=mlcontests),
[Tailwind CSS](https://tailwindcss.com/?ref=mlcontests), [Chart.js](https://www.chartjs.org/?ref=mlcontests),
[Linguist](https://github.com/github/linguist?ref=mlcontests), [pipreqs](https://github.com/bndr/pipreqs?ref=mlcontests), and
[nbconvert](https://github.com/jupyter/nbconvert?ref=mlcontests).

### Methodology

Data was gathered from correspondence with competition platforms, organisers, and winners, as well as from public
online discussions.

The following criteria were applied for inclusion, in-line with the [submission criteria](https://mlcontests.com/submit?ref=state24) for our
listings page. Each competition must:

1. have a total available prize pool of at least $1,000 in cash or liquid cryptocurrency, or be an
official competition at a relevant conference.
2. have ended between 1 Jan 2024 and 31 December 2024 (inclusive).

Where possible, we used data provided by each competition platform, describing their own competitions, as a
starting point. Most platforms were able to provide us with this data for 2024.

We did not receive this data from all platforms, and in these cases we gathered the data from their website. Notably,
for CodaLab, Codabench, and EvalAI, three ‘self-service’ platforms where competition organisers can create their own
competitions with minimal to no intervention from the platform team, we did our best to collect complete data and filter
out irrelevant competitions, such as ones used for class assignments or draft competitions which never ended up running.

Our filter for ‘relevant conferences’ was broader this year, as we started with a superset of potentially relevant
competitions from competition platforms before filtering irrelevant ones out, and found a long tail of conferences with
interesting ML-related competitions. This likely contributed significantly to us finding more relevant competitions
without cash prizes.
In future years, we may be more selective with our conference filter and have a
short list of included conferences. We may also choose to exclude conference workshop competitions in future, including
only competitions for conferences with a separate competition track.

We were not able to collect full data for all platforms. For example, the dsworks.ru site bore a
notice stating that the team had _“temporarily taken our website offline for updates”_ when we tried to access it.
It is likely that there are additional competition platforms as yet unknown to us. Please contact us if you know of any;
we are open to including them in future.

When counting a “number of competitions” for purposes such as prize pool distribution, or popularity of programming
languages, we generally use the following definition:

_If a competition is made up of several tracks, each with separate leaderboards and separate prize pools, then each track_
_counts as its own competition. If a competition is made up of multiple sub-tasks which are all measured together on_
_one main leaderboard for one prize pool, they count together as one competition._ There are some exceptions. [7](https://mlcontests.com/state-of-machine-learning-competitions-2024#fn:7)

For the purposes of this report, we consider a “competition winner” to be the #1-placed team in a competition as defined above.
We are aware that other valid usages of the term exist, with their own advantages — for example, anyone winning a Gold/Silver/Bronze medal,
or anyone winning a prize in a competition. For ease of analysis and in order to avoid double-counting, we exclusively
consider #1-placed teams in this report when aggregating statistics on “winning solutions” or “competition winners”.

Compiling the [Python packages](https://mlcontests.com/state-of-machine-learning-competitions-2024#python-packages) section in the winning toolkit involved some discretion. While we
attempted to highlight the most popular and interesting packages for readers, we did not simply take the n most popular
packages.

The number of users for each platform is sourced from the platform directly, where possible. Some platforms, like
Kaggle, AIcrowd, CodaLab, and Codabench, list their user numbers publicly on their website. Most other platforms shared
their user numbers with us via email. Hugging Face does not have user accounts specific to its competitions product.
Solafune chose not to disclose their user numbers. User numbers for Signate, Tianchi, and Bitgrit are a year old, as we
did not receive updates on their user numbers before publication of this report.

We excluded the DARPA AI Cyber Challenge small business track, as it wasn’t open to individuals. We excluded the
AI Agents Global Challenge, as the prizes were a mixture of investment and compute credits, not cash. We excluded the
Google Gemini API Developer Competition, as we felt it was more of a product development competition than a machine
learning competition.

List of packages


```
[\
 ('numpy', 63),\
 ('pandas', 58),\
 ('torch', 54),\
 ('tqdm', 42),\
 ('sklearn', 35),\
 ('matplotlib', 26),\
 ('transformers', 25),\
 ('scipy', 24),\
 ('lightgbm', 16),\
 ('seaborn', 14),\
 ('cv2', 14),\
 ('catboost', 13),\
 ('torchvision', 12),\
 ('timm', 11),\
 ('joblib', 11),\
 ('pil', 11),\
 ('yaml', 10),\
 ('peft', 9),\
 ('albumentations', 9),\
 ('tensorflow', 9),\
 ('requests', 8),\
 ('datasets', 8),\
 ('loguru', 8),\
 ('xgboost', 8),\
 ('optuna', 7),\
 ('polars', 7),\
 ('accelerate', 6),\
 ('trl', 6),\
 ('wandb', 6),\
 ('einops', 6),\
 ('langchain', 5),\
 ('openai', 5),\
 ('skimage', 5),\
 ('sentence_transformers', 4),\
 ('rasterio', 4),\
 ('typer', 4),\
 ('shapely', 4),\
 ('keras', 4),\
 ('setuptools', 4),\
 ('beautifulsoup', 3),\
 ('omegaconf', 3),\
 ('kornia', 3),\
 ('click', 3),\
 ('geopandas', 3),\
 ('dataretrieval', 3),\
 ('zeep', 3),\
 ('pydantic', 3),\
 ('torchmetrics', 3),\
 ('pytorch_lightning', 3),\
 ('librosa', 3),\
 ('simpleitk', 3),\
 ('nibabel', 3)\
 ]

```

### Attribution

For attribution in academic contexts, please cite this work as

```
Carlens, H, “State of Machine Learning Competitions in 2024”, ML Contests Research, 2025.
```

BibTeX citation

```
@article{
carlens2025state,
author = {Carlens, Harald},
title = {State of Machine Learning Competitions in 2024},
journal = {ML Contests Research},
year = {2025},
note = {https://mlcontests.com/state-of-machine-learning-competitions-2024},
}
```

* * *

01. Over half of the total prize money in 2024 was for the DARPA AI Cyber Challenge. Excluding this
    competition, the total available prize pool of around $8.4m was still higher than in 2022 and 2023. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:1)

02. The majority of CodaLab’s competitions in 2024 were conference workshop competitions, such as
    for the [NTIRE](https://cvlai.net/ntire/2024/?ref=mlcontests) workshop at CVPR, with no prize money associated.
    Currently these competitions are included in our report dataset (given that our criteria state
    “conference affiliation”), but we are in the process of reviewing our inclusion criteria and may in future reports
    restrict the criteria to include only conference competitions that are part of official competition tracks, such as at
    NeurIPS or ICRA, or with a meaningful monetary prize. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:2)

03. As per the [CodaLab newsletter](https://github.com/codalab/codabench/wiki/%5BNewsletter%5D-CodaLab-in-2024#-introducing-codabench?ref=mlcontests):
    “Codabench platform software is now concentrating all development effort of the community. In addition to CodaLab
    features, it offers improved performance, live logs, more transparency, data-centric benchmarks and more!
    We warmly encourage you to use codabench.org for all your new competitions and benchmarks.” [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:3)

04. Public user numbers taken from [AIcrowd](https://www.aicrowd.com/?ref=mlcontests),
    [CodaLab](https://codalab.lisn.upsaclay.fr/highlights?ref=mlcontests),
    [Codabench](https://www.codabench.org/?ref=mlcontests),
    [EvalAI](https://eval.ai/?ref=mlcontests),
    [Grand Challenge](https://grand-challenge.org/stats/?ref=mlcontests),
    [Kaggle](https://kaggle.com/?ref=mlcontests),
    and [Zindi](https://zindi.africa/?ref=mlcontests)
    on 18 February 2025.
    User numbers for Bitgrit, Signate, and Tianchi are a year old, as we weren’t able to get an updated figure this year.
    For other platforms, user numbers were provided by the platform team over email. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:4)

05. The number of competitions and total prize money amounts are for competitions that ended in 2024.
    Prize money figures include only cash and liquid cryptocurrency. Travel grants and other types of prizes are excluded.
    Amounts are approximate — currency conversion is done at data collection time, and amounts are rounded to the
    nearest $1,000 USD. See [Methodology](https://mlcontests.com/state-of-machine-learning-competitions-2024#methodology) for more details. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:5)

06. User numbers for Bitgrit are as of March 2024. We reached out to the team at Bitgrit for updated user
    numbers but did not get a response. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:6)

07. CrunchDAO’s [DataCrunch](https://hub.crunchdao.com/competitions/datacrunch?ref=mlcontests)
    competition is an ongoing US equity market prediction competition, where prizes are paid out monthly and a total of
    $120k is available per year. The CrunchDAO website states that it started on 25/06/2024. In our dataset we have counted
    this as one competition with a $60k prize pool in 2024. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:7) [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref1:7)

08. “More than 80,000 high-level human resources are participating, including excellent data scientists at major companies and students majoring in the AI field. (as of February 2023)”. Source: [Signate’s website](https://signate.jp/company_about?ref=mlcontests), translated from Japanese using Google Translate on 5th of March 2024. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:8)

09. Tianchi user numbers are from April 2024. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:9)

10. Anecdotal evidence from speaking to multiple quant trading firms at ML conferences who explicitly
    stated that they value ML competition performance in applicants, as well as competitors who have been contacted by quant
    recruiters because of their position on certain competition leaderboards. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:10)

11. When analysing winning solutions, we restrict to competitions where ranking is based fully or
    primarily on modelling performance, excluding competitions based around writing, data visualisation, or other elements
    that require subjective ratings from a panel of judges. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:11)

12. Out of the 53 competition winners whose compute resources we were able to confirm, 44 (83%) used
    NVIDIA GPUs, 8 only used CPU resources, and one used a Google TPU. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:12)

13. Tens of thousands of research papers published in 2024 cited the use of NVIDIA chips, whereas only
    hundreds cited AMD chips, as per the [State of AI Report Compute Index](https://www.stateof.ai/compute?ref=mlcontests). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:13)

14. SemiAnalysis: [MI300X vs H100 vs H200 Benchmark Part 1: Training – CUDA Moat Still Alive](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/?ref=mlcontests). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:14)

15. The [median hourly on-demand H100 price is around $3](https://cloud-gpus.com/?ref=state24)
    at the time of writing. Lambda Cloud, which was used by the winners of the ARC challenge, [charges $2.99/h/H100 for an\\
    8xH100 node](https://lambdalabs.com/service/gpu-cloud?ref=mlcontests#pricing). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:15)

16. Because we don’t have a reliable way to link user accounts across competition platform sites,
    these figures only count repeat wins on a given platform. Someone who wins a competition on Kaggle and later wins one on
    DrivenData will be counted as a first-time winner in the second competition. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:16)

17. The competition rules required the use of Falcon-7.5B or Phi-2, but the choice of ColBERT was down to
    the winners themselves. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:17)

18. The “Tuning Meta LLMs for African Language Machine Translation” required that
    “Winning solutions must use open-source models from the Meta repository on Hugging Face, including,
    but not limited to, NLLB 200, SeamlessM4T, as well as any derivatives of these models.” [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:18)

19. For example, during training for the Llama 3 models, parameters were mostly stored in bfloat16 and
    gradients in float32:
    _“To ensure training convergence, we use FP32 gradient accumulation_
    _during backward computation over multiple micro-batches and also reduce-scatter gradients in FP32 across_
    _data parallel workers in FSDP. “_ ( [paper](https://arxiv.org/abs/2407.21783)).
    More recently, DeepSeek V3 used mixed-precision training with FP8: _“we cache and dispatch activations in FP8, while_
    _storing low-precision optimizer states in BF16”_ ( [paper](https://arxiv.org/pdf/2412.19437v1))
    For an introduction to mixed-precision training,
    see [Sebastian Rashka’s overview from May 2023](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:19)

20. The bfloat16 format has the same range as the 32-bit float format, but with lower precision (whereas the
    normal 16-bit float format sacrifices both range and precision as compared to float32). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:20)

21. As per the [Polars blog](https://pola.rs/posts/announcing-polars-1/?ref=mlcontests): _“With this release, we signify that the Polars in-memory engine and API is production ready.”_ [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:21)

22. The paper claims “6 gold medals, 3 silver medals, and 7 bronze medals, as defined by
    Kaggle’s progression system”, however:


    - None of the submissions were for active competitions; all submissions were for competitions that had already ended.
    - All the claimed “gold” and “silver” medals, as well as 5 of the “bronze” medals, were for _Community_,
      _Getting Started_, or _Playground_ competitions. These types of competitions do not award medals.
    - The two other claimed “bronze” medals, one for a _Featured_ competition and the other for a _Research_ competition,
      were for competitions which ended in 2015 and 2016.

[↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:22)
23. On p. 38 of their paper they state a caveat that downplays their claims somewhat:
    _“Finally, while we refer to our agent as having “Kaggle Grandmaster-level”_
    _performance, we clarify that this does not imply a formal Grandmaster title on Kaggle._
    _Our achievements include several gold medals across various competition_
    _types — community, playground, and featured — demonstrating our agent’s competitive capability among data scientists,_
    _but some competitions may be less complex than others, and some are non-medal awarding competitions._
    _To formally pursue the Grandmaster title, we aim to focus on active competitions, scaling and improving our results as_
    _we advance to newer versions of Agent K.”_ [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:23)

24. The first six distilled versions of DeepSeek R1 [were whitelisted](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/548129#3103634)
    on Thu 23 Jan 2025 at 18:17 GMT. Our “pre-whitelisting” leaderboard snapshot was taken around three hours earlier,
    at 15:14 GMT on the same day. Our post-whitelisting leaderboard snapshot was taken at 10:44 GMT on Mon 27 Jan. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:24)

25. The top-performing solution on the private leaderboard was by team MindsAI, with a score of 55.5%.
    Team MindsAI chose not to open-source their solution, thereby foregoing the prize money.
    Team MindsAI is made up of Jack Cole and Mohammed Osman (joint winners of the [2023 ARCathon](https://lab42.global/winners/?ref=mlcontests)),
    and Michael Hodel (winner of the 2022 ARCathon).
    All three MindsAI members are now part of [Tufa Labs](https://tufalabs.ai/?ref=mlcontests), who describe themselves
    as “a small, independent research group working on fundamental AI research” initially focused on completing the ARC Prize challenge. [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:25)

26. For more on depth-first search and other meta-generation strategies that can be used to enhance
    language model performance, see the NeurIPS 2024 tutorial
    [Beyond Decoding: Meta-Generation Algorithms for Large Language Models Beyond Decoding: Meta-Generation Algorithms for Large Language Models](https://cmu-l3.github.io/neurips2024-inference-tutorial/?ref=mlcontests) [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:26)

27. In their o3 blog post, the ARC Prize team note that _“OpenAI shared they trained the o3 we tested on 75%_
    _of the Public Training set. They have not shared more details. We have not yet tested the ARC-untrained model to_
    _understand how much of the performance is due to ARC-AGI data.”_
    For more details, see the ARC Prize’s blog posts on [o3](https://arcprize.org/blog/oai-o3-pub-breakthrough)
    and [R1](https://arcprize.org/blog/r1-zero-r1-results-analysis). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:27)

28. Andrew Carney quote taken from the [DARPA blog](https://www.darpa.mil/news/2024/ai-cyber-challenge-cybersecurity). [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:28)

29. Vesuvius Challenge blog,
    [First letters found in new scroll](https://scrollprize.substack.com/p/first-letters-found-in-new-scroll) [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:29)

30. [https://scrollprize.substack.com/p/exciting-news-from-scroll-5](https://scrollprize.substack.com/p/exciting-news-from-scroll-5) [↩︎](https://mlcontests.com/state-of-machine-learning-competitions-2024#fnref:30)