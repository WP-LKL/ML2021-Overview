
Summary of the 70 page cookbook with code and equations omitted in an easily shareable format.

# Machine Learning Tasks
```
Initialize -> y_hat := Predict(activation(W * X)) 
           -> Loss(y_hat, y)
           -> W.Gradient(Loss)
           -> Optimizer(W, learning_rate) 
           -> y_hat ...
```
## NLP (Natural Language Processing)
### Tasks
* Seq2Seq (Encoder Decoder)
    * Encoder: Embedding, Hidden, 
* Seq2Token
* Token prediction
### Architecture
* RNN (Recurrent Neural Networks)
    * Vanishing/Exploding gradient problem
    * Beam search / Greedy Search
* LSTM (Long Short Term Memory)
    * Difficult to train
    * I/O bottleneck - poor parallelization
        * stnd(a) := W_a @ x_t + U_a * h_{t-1} + b_a
        * h_t <- o_t * tanh(c_t)                  Hidden state, immediatly necessary information to next layer
        * c_t <- f_t * c_{t-1} + i_t * j_t        Contextual longer term information
        * o_t <- sigm(stnd(o))                    Cell state
        * f_t <- sigm(stnd(f))                    Ouput gate
        * i_t <- sigm(stnd(i))                    Input gate
        * j_t <- tanh(stnd(j))                    Modulation gate
        * U: m x m == hidden x hidden
        * W: m x d == hidden x dims
        * Computational order:
        * <f, i, j, o>  -->  c  -->  h 
        
* Transformers with Transfer Learning
    * HuggingFace
    * Gene

## Time-Series data
* RNN (Recurrent Neural Networks)
    * Vanishing/Exploding gradient problem
* HMM (Hidden Markov Models)
    * Poor parallization
    * Relies on domain knowledge
    * Less reliant on training data 

## Computer Vision
* CNN (Convolutional Neural Networks)

## Tabular data
* SVM (Support Vector Machines)
    * Valuable niche kernels
* Decision Trees (Random Forest)
    * Ensemble of full trees
* Gradient Boosting
    * Ensemble of weak trees

## Unsupervised Clustering
* AP (Affinity Propagation)
    * Noise sensitive
* K-NN (K-Nearest Neighbors)
    * Predefined number (K) of clusters
* DBSCAN
    * Excels with noise-less data

## Unsupervised Simulation
* GA (Genetic Algorithms)
    * Big discrete search space
* RL (Reinforcement Learning)
    * Greedy problems

## Other Unsupervised tasks
* Generative Adversarial Networks


## Semi-Supervised (Partially labelled)
* Co-Training
* Self-Training
* Tri-Training

# Data Engineering
``EDA -> Transform -> Classification -> Evaluate``
1. EDA: Heatmaps, Missing values, Distribution
2. Transform Data: Scale, Standardize, Normalize, Drop rows with missing values, Balance dataset, Impute, rotate/mirror, add noise




For encoding and scaling remember to fit these feature-maps exclusively on the training data.

## Categorical Data
* Onehot encoding
    * High cardinality data lead to sparseness 
    * Not great for Tree based classifiers
* Truncation
    * Truncate similar categories
    * Remove/Truncate infrequent categories
    * Can often be deployed along with other techniques
* Binary (base-N) encoding
* Backwards Difference encoding
* (and many others..)

## Missing Value Imputation
* Mean imputation
* Similarity imputation

## Text Representation
* BOW (Bag of Words)
    * Each sentence is a vector of words
    * No context, scales poorly
    * Weighted TF-IDF (uniqueness) representation
* W2V (Word 2 Vector)
    * Each word is a vector
    * Good for dynamic vocabulary


## Continous Data 

* Logarithmic transform
    * Modulo: ``sign(x) * log1p(abs(x))``
    * Decreasing the distance between linear data with binary characteristics
* Min/Max Scaler
    * Outliers Skew the distribution
    * Suitable for Neural Networks
    * Truncates information with outliers.
* Standard Scaler
    * Outliers Skew the distribution
    * Centered staggered mean

Statistics come in to play when picking scalers. Knowing the distribution of your inputs will help you succeed when engineering your features. For this purpose try plotting a histogram of your values.

### Examples:
#### <b> Right-tail Income/spending data:</b>

Often there is a nonlinear relationship between income/spending and a given output say task about predicting spending on rent or customer importance for a business. By shifting the distribution using a logaritm, future outliers can also be contained. Specifically, one might apply multiple transformations:

``Log1p()  -->  MinMax/StdScaler/RobutScaler``

If you are dissatisfied with the Natural Log transform you can of course try different log transformations for softer or heavier regularization. (Understanding the task is key here.)

## Dimensionality reduction
* PCA (Principal Component Analysis)
    * Convert features to high-variance Principal Components
* Clustering
* Filtering
    * Low Variance filter
    * Correlation Filter
* Variational Autoencoders

## Other
* Cyclical data: Month/Time/Day
    * Sin/Cos Relationship


## Train/Test Split

### Entity data
A lof of medical data or image detection revolves around an entity. A shortcut for the model is to learn the person and infer the resulting output from that person, but when encountering a new person the model will fail miserably. It is very important to have the test data simulate real world application, so separate persons in the training and testing data set. (No cross-validation!)

### Timeseries data
>"Wow! So you achieved 20% YOY when "backtesting" -- that is literally a money printer, let me put all my savings into it!"

For timeseries data always set aside the last x months/weeks/days for a more robust approximation of real world performance for the test set. No cross-validation here either.


# Neural Network Architecture
## Kaggle meta
* Complex models with heavy regularization
    * BatchNorm - Regularize inter-layer momentum
    * Dropout - Dependency shooting, more robustness
* Blending (Rough Ensemble)

### CNN    
* 2 stride kernels over pooling.

## State of the art

### CNN
* 2x2 Average pooling layer before entering the linear NN.
* Repeated use Components/Blocks consisting of parallel capture (revisiting).
    * Inception, Identity, Conv[A,B,C], Reduction...

# Auxiliaries
## Loss / Objective function
Non-convex vs Convex loss surface: Random initialization --> Local vs Global minima/maxima.
* MSE (Mean Squared Error)
    * Punishes very deviant errors heavily due to squaring
    * Is not concerned with direction
* MAE (Mean Absolute Error)
    * More robust against outliers
    * Is not concerned with direction
* MBE (Mean Bias Error)
    * Errors are fine, as long as they cancel out
* Huber loss / Smooth mean Absolute error
* Pseudo Huber Loss
* Log cosh Loss
* Quantile Loss
* Hinge Loss
* Log Loss (Cross Entropy)
    * Penalizes heavily confident predictions that are wrong
    * Probabilities between 0-1
* Focal
* Exponential
* KL Divergence / Relative Entropy


## Evaluation
E_test (test error) only resembles accurately E_out (unseen error) when we are not tuning (fitting) for the test set by picking based off multiple iterations/models on the test set. 

|E_in - E_test|, that is how much do we overfit the training data.

|E_test - E_out|, is how much we contaminated and used our test set for determining generalization. But after E_test affects which models we select for retraining, we end up contaminating the resemblance of E_test to E_out. Imagine tinkering with hyperparameters for the model that achieves the best E_test, in this case we have fitted our model to the test set, and its purpose for E_out falls to the ground. In this scenario we can only talk about expected real world performance by introducing a completely unseen validation set. A probabilistic bound can often be derived, such as Hoeffding for balanced datasets. 

E_in : Training error.

|E_in-E_val| : Model Complexity and overfit.

|E_val-E_test| : Validation Contamination.

E_test : Real-world performance proxy. 

---
Robustness gap: Pr[correct on x_i] - Pr[correct on x_i  |  5% noise ]

Rationality gap: Pr[correct on x_i  |  5% noise, y noisy] - Pr[correct on x_i  |  fresh x ]

Memorization gap: Pr[correct on x_i  |  5% noise] - Pr[correct on x_i  |  5% noise, y noisy]

---
* Cross-validation
    * Use more data with reasonable error accuracy
    * More folds --> more training time & better estimate
    * We can compute model fragility
    * Avoid on Timeseries and Entity data!
* ROC/AUC
    * False positives and false negatives
    * Visualization particularly for binary classifiers
* F score
* Confusion Matrix

## Optimizer
* Stochastic GD
    * Slow but solid results
    * Improved with batches or Nesterov momentum
* ADAM
    * 2nd order momentum
    * Fast, but tends to overfit

## Activation
Computing gradients, we need distinct slopes that don’t require 2^32 bit floats as with sigmoid and other rounded activation functions near the low gradients. We want our activation function to have a preferably easily computable derivative.
* ReLU
    * Dead Neurons
* ELU
* Leaky ReLU
    * Discontinous origin

## Kernels (SVMs)
Mapping your data to a dimension where it is linearly separable. With a Gaussian kernel we map the relation between data entries with respect to each other following a gaussian distribution. The variance parameter of the kernel then describes the shape of the gaussian and hence the bias-variance tradeoff.

* RBF (Radial Basis Functions)
* Legendre Polynomial
    * Convex, double differentiable, and invertible properties
* Polynomial Kernel

## Faster training
* Use 16/24 Floating Point (Nvidia Apex)
* Evaluate and interpret Loss vs (Validation) Error improvements and consider deploying early-stopping

# AutoML
## Regression & Classification
* Auto-Sklearn [Good Baseline]
* TPOT (Tree-based Pipeline Optimization Tool)
* H2O
## Neural Networks
* ADAnet [Not recommended]
    * Uses components
## HO (Hyperparameter Optimization)
* Wandb Sweep [Recommended]
    * Distributed systems
    * Bayesian Optimization
    * Persistent logging
* Hyperopt-Sklearn
    * Sklearn
    * Bayesian Optimization
* GridSearch
    * Simple configuration



# Developer tools
## Meta Data and Monitoring
|            | Hosted |
|------------|--------|
| WandB      | Yes    |
| ClearML    | Yes    |
| Neptune.ai | Yes    |
| Comet.ml   | Yes    |

Others: Pachhyderm, Valohai, Sacred (Omniboard), GuildAI, MLflow, Tensorboard, DVC, Verta, Polyaxon, Trains.
## Pipeline/ETL
|                | Developed |
|----------------|-----------|
| Apache Airflow | Airbnb    |
| Luigi          | Spotify   |
| Prefect        |           |
| Kedro          |           |
| DBT            |           |
| Dagster        |           |
| Databricks     |           |
| Snowflake      |           |


Beyond these most cloud platforms will have their own scheduler and tools for Extracting, Transforming, and Loading.

## Notebooks
|          | Stability | Accesability | Quota | Cores | RAM  | GPU  | TPU | Disk | Persistent | Collaboration |
|----------|-----------|--------------|-------|-------|------|------|-----|------|------------|---------------|
| Kaggle   | Poor      | Poor         |   540 |       | 16GB | GPU  | TPU | 20GB |            |               |
| Colab    | Mediocre  | Good         |       |       | 12GB | GPU  | -   | 75GB | -          | Delayed       |
| cocalc   |           |              |       |     1 | 1GB  | -    | -   | 3GB  |            | Realtime      |
| IBM DS   |           |              |       |       |      |      |     |      |            |               |
| Datalore |           |              |       |     2 | 4GB  | Paid | -   |      | Partial    |               |
| DeepNote | Mediocre  | Great        |       |       |        | Paid |     | 5GB  |            | Realtime      |

Cloud platforms (Azure, Amazon, Google, Alibaba..) also offer Notebooks.

# Contributing
Highly valuable condensed information and corrections are greatly appreciated.