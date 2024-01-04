# Collaberative-Filtering-Recommendation-Systems
[In Depth Walk Through / Blog Post](https://medium.com/@justin.donnelly0804/using-machine-learning-to-identify-twitter-spam-acdc05e78b15)

This project creates a model to recommend books/movies based on learned latent factors. This is done using collaberative filtering.The following model was created and trained to learn latent factors about users and books from how they rated the books:

```python
class DotProductBias(Module):
    def __init__(self, n_users, n_books, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users,n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.book_factors = Embedding(n_books,n_factors)
        self.book_bias = Embedding(n_books,1)
        self.y_range = y_range
    
    def forward(self,x):
        users = self.user_factors(x[:,0])
        books = self.book_factors(x[:,1])
        residual = (users*books).sum(dim=1, keepdim=True)
        residual += self.user_bias(x[:,0]) + self.book_bias(x[:,1])
        return sigmoid_range(residual, *self.y_range)

