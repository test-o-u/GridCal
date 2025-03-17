from model import Model
from symprocess import generate, generate_pyc

object1 = Model()
object2 = Model()

filename = generate("object1", object1)
generate_pyc(filename)

print("Processing complete! Python files, compiled .pyc files, and .npz cache files are ready.")