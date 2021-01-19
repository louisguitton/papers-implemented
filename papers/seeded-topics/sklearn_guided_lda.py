"""

Ref:
-  Li et al (2017)
- https://stackoverflow.com/questions/45170093/latent-dirichlet-allocation-with-prior-topic-words#45170093

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.
    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.
    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.
    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.
"""
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import check_random_state
from sklearn.decomposition._lda import _dirichlet_expectation_2d
import numpy as np
from scipy.sparse import csr_matrix


class GuidedLatentDirichletAllocation(LatentDirichletAllocation):
    # TODO: check what value to use from the papers
    _SEED_WEIGHT = 100.

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
        seed_words:List[List[str]]=None,
        tf_vectorizer: CountVectorizer=None
    ):
        super(GuidedLatentDirichletAllocation, self).__init__(
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

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        # ##############
        # Start sklearn code
        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features))
        # End sklearn code
        # ##############
        # Start custom code

        # Transform topic values in matrix for prior topic words
        seed_documents: List[str] = [" ".join(topic_words) for topic_words in self.seed_words]
        topics_to_words: csr_matrix = self.tf_vectorizer.transform(seed_documents)
        ones = np.ones(self.components_.shape)
        ones[:topics_to_words.shape[0], :topics_to_words.shape[1]] += topics_to_words * self._SEED_WEIGHT
        self.components_ = np.multiply(self.components_, ones)

        # End custom code
        # ##############
        # Start sklearn code
        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_))
