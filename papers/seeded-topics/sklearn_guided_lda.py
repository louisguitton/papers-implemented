"""

Ref:
- Jagarlamudi, J., Iii, H. D., & Udupa, R. (2012). Incorporating lexical priors into topic models.
    EACL 2012 - 13th Conference of the European Chapter of the Association for Computational
    Linguistics, Proceedings, 204–213. https://www.aclweb.org/anthology/E12-1021
- https://www.freecodecamp.org/news/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164/
- https://stackoverflow.com/questions/45170093/latent-dirichlet-allocation-with-prior-topic-words#45170093
"""
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import check_random_state
from sklearn.decomposition._lda import _dirichlet_expectation_2d
import numpy as np
from scipy.sparse import csr_matrix


class GuidedLDA(LatentDirichletAllocation):

    def __init__(
        self,
        n_components=10,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1000000.0,
        perp_tol=0.1,
        mean_change_tol=0.001,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
        seed_words: List[List[str]] = None,
        seed_confidence: float = 100.,
        tf_vectorizer: CountVectorizer = None,
    ):
        super(GuidedLDA, self).__init__(
            n_components,
            doc_topic_prior,
            topic_word_prior,
            learning_method,
            learning_decay,
            learning_offset,
            max_iter,
            batch_size,
            evaluate_every,
            total_samples,
            perp_tol,
            mean_change_tol,
            max_doc_update_iter,
            n_jobs,
            verbose,
            random_state,
        )
        self.seed_words = seed_words
        self.tf_vectorizer = tf_vectorizer
        self.seed_confidence = seed_confidence

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        # ##############
        # Start sklearn code
        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features)
        )
        # End sklearn code
        # ##############
        # Start custom code

        # Transform topic values in matrix for prior topic words
        seed_documents: List[str] = [
            " ".join(topic_words) for topic_words in self.seed_words
        ]
        topics_to_words: csr_matrix = self.tf_vectorizer.transform(seed_documents)
        bias = np.ones(self.components_.shape)
        bias[
            : topics_to_words.shape[0], : topics_to_words.shape[1]
        ] += topics_to_words * (self.seed_confidence)
        self.components_ = np.multiply(self.components_, bias)

        # End custom code
        # ##############
        # Start sklearn code
        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
