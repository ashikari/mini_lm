# mini_lm
Basic repository to learn how to pre-train an LLM

## Dataset Selection

For this project, we have chosen to use the BabyLM dataset instead of the more commonly used WikiText-103 dataset. While WikiText-103 is a popular choice for language model training, its large size (103 million tokens) and complex vocabulary make it less suitable for training a mini LLM. The formal Wikipedia writing style and technical content can also introduce unnecessary complexity for initial language model training experiments.

Our choice of BabyLM is based on several key advantages:

1. **Developmentally Plausible Training Data**: BabyLM provides a dataset that more closely mirrors the linguistic environment of a child's development, making it particularly suitable for studying language acquisition and model development.

2. **Controlled Vocabulary**: The dataset is carefully curated to include developmentally appropriate English language patterns and vocabulary, which can lead to more interpretable model behavior. The focus on English-only content ensures consistency and reduces potential confusion from multilingual elements.

3. **Quality and Diversity**: While maintaining a developmentally appropriate scope, the dataset still provides sufficient diversity and quality for effective language model training.

For more details about the BabyLM dataset and its design principles, please refer to the original paper: [BabyLM: A Developmentally Plausible Training Corpus for Language Models](https://arxiv.org/pdf/2301.11796v1)
