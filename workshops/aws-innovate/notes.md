## Opening keynote

We're on top of all topics.
They emphasised many things but for us mainly AWS Forecast and AWS Personalize are interesting

## Scale ML from 0 to million users - DS track

https://gitlab.com/juliensimon/dlcontainers
https://gitlab.com/juliensimon/dlnotebooks

1. run everything on your laptop
2. run in the cloud on an EC2 instance using an AMI instance
   demo on AMIs
3. scaling issues
   no IaC (infrastructure as clicks) => write code + CI/CD
   option 1 : continue with virtual machines but a pain in the ass because of the overhead
   option 2 : docker clusters ECS or EKS => good because you share infra wiht the team and use docker like them
   demo on ECS
   option 3 : go fully managed with AWS Sagemaker => no infrastructure to manage

Recommenders on SageMaker https://gitlab.com/juliensimon/dlnotebooks/blob/master/sagemaker/03-Factorization-Machines-Movielens.ipynb

go away from NIH or Hype driven development: k8s has not been designed for ML in mind, so no point in going for that if that's not what your business is.
It should be a red flag if you notice that you spend 80% of your time doing infrastructure and not ML

interesting that fargate is mentioned but not delved into

## Advanced SageMaker - DS Track

Notebook instances are great

3 options:

- 17 off the shelf algos where you bring nothing => pure sagemaker python sdk
- 5 frameworks where you bring your code
- container mode where you bring your environment => mlflow

2 sdks:

- sagemaker python sdk for ML stuff
- aws sdk / boto3 for devops stuff

pipe mode: training data can be in S3 or EFS and doesn't get replicated to the training instance

demo : word embeddings
https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia

really need to salvage my spacy word embeddings using blazing text
and see how to use that in the new rule based model

check out Gluon NLP on top of Apache MXNet
https://gluon-nlp.mxnet.io/

## Deep dive Sagemaker

Storage

- pipe mode for tensorflow demo = good for large datasets (20 Gb)
- FXs => too advanced for us

Training

- DistributedTraining mode
  - horovod by Uber (efficient algo to have better distributed training performance)
  - EC2 P3dn (new monster instance)
  - managed spot instances training => save money
- Use potobuf and RecordIO for file formats

Tuning

- instead of manual search or grid search or random search; use Hyperparmeter Optimisation (HPO)

Deployment

- optimise for cost because inference can be expensive => use Neo to get a 7x improvement
- right size your inference infra => AWS Elastic Inference

## Forecast

- choose AutoML
- DeepAR+

- BackTestWindow specific to time series instead of regular 80-20 split

- look into quantile loss and not RMSE because onw bad week amplifies the error otherwise
- get p10 p50 and p90

## Personalize

- personalised recommendations
- personalised notifications

* need to build the items dataset with blazing text embeddings for metadata
* for datasets they use Avro schema
* for metrcis: is order important for our use case ? if no: precision; if yes: normalised discounted cumulative gain

* recording live events
  from the backend with python (create a sidecar backend service with ambassador)
  from frontend with aws amplify JS

## Closing words

with Data Scientist at Expedia, also AWS ML Hero: he spends 90% of his time doing data engineering and 10% ML

- the most important thing to be successful at big AND small companies is to have a good ML platform (sagemaker, mlflow, ...)
- rely on off the shelf models
- xgboost is still winning on kaggle
- find an informal mentor that has done ML for many years in the workplace (go to mlbox)

- https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow-sagemaker
- https://mlflow.org/docs/latest/models.html#sagemaker-deployment


## Next steps

Immediate:
- prepare word-embeddings dataset into text8 format, get at least 100 MB of data per language

> BlazingText expects a single preprocessed text file with space separated tokens and each line of the file should contain a single sentence.

- train blazingtext word embeddings in SageMaker (=salvage my spacy work + will be useful for metadata in the Items dataset of Personalize + will be useful to compute word similarities if that's needed in the rule based models)
    - https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/blazingtext_word2vec_subwords_text8
    - https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/blazingtext_word2vec_text8
- prepare textcat dataset in fasttext format
- train textcat in SageMaker
    - https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia
- adapt commonness and TAGME to rank entities based on tokens, useful for part 2 of the rules based architecture

Future:
- try https://gitlab.com/juliensimon/dlnotebooks/blob/master/sagemaker/03-Factorization-Machines-Movielens.ipynb
