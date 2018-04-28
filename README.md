# A Weakly Supervised Graph-based System for Customer Review Categorization

As Web has become one of the most important sources for customers to evaluate and compare products and
services, the providers want to monitor the opinion of their costumers with respect to a set of aspects such as price,
service and quality. Hence, automated ways to handle and classify customer reviews have been widely studied in the
area of Opinion Mining. Most of the studies use supervised classification methods on a manually annotated dataset.
Although supervised systems obtain good results for the domain they are trained on, they are almost useless when
the domain or language has changed. Annotating data for all domains and languages is not applicable. Because of
that, recent studies have focused on unsupervised or semi-supervised methods for customer review categorization
task.

In this project, the goal is to develop a system which is able to automatically categorize customer reviews into
aspect categories defined by the users. The proposed system is almost unsupervised and requires only a couple of
reviews labeled with an aspect category. With a little effort, users are able to explore and analyze customer reviews
in any domain.

In the first step of proposed system, the reviews will be tokenized and lemmatized, the stopwords will be removed to
avoid data sparsity problem. Then each review will be represented with a dense vector by leveraging pre-trained word
vectors (a.k.a word embeddings) [1] and paragraph vectors (a.k.a doc2vec) [2]. A similarity score will be calculated
for each review pair using cosine distance between vector representations of reviews. Using these similarity scores
the reviews will be represented with a graph where nodes are reviews and edges are similarity scores. (A threshold
parameter will be used to cut off the edges which have similarity scores less than the threshold value). For each userdefined
aspect category, Personalized PageRank algorithm [3] will be performed on similarity graph using labeled
reviews as topic specific nodes. By this way, each review node will be scored for each aspect category.

This project is carried out as a term project in a PhD Course (Web Mining ITU BLG 614E) in Istanbul Technical University See detailed report in Web Mining Project Report for more detail explaation and experimental results.
