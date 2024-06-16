import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i]
        left_side = np.dot(matrix, eigenvector)
        right_side = eigenvalue * eigenvector

        if np.allclose(left_side, right_side):
            print(f"Eigenvalue {eigenvalue} satisfies equation")
        else:
            print(f"Eigenvalue {eigenvalue} and {eigenvector} do not satisfies equation")

    return eigenvalues, eigenvectors


A = np.array([
    [1, 1, 0],
    [9, 2, 0],
    [0, 1, 1]
])

eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(A)
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Second part
# initial_image = np.asarray(Image.open("The-person-who-thinks-all-the-time....png"))
# image_bw = np.asarray(initial_image[:, :, 0])
# plt.imshow(image_bw, cmap='gray')
# plt.show()
#
# pca = PCA()
# image_pca = pca.fit_transform(image_bw)
# image_reconstructed = pca.inverse_transform(image_pca)
#
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#
# plt.plot(cumulative_variance)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Cumulative Explained Variance vs Number of Components')
# plt.grid()
# plt.show()
#
# n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
# print(f'Number of components to cover 95% variance: {n_components_95}')
#
# pca = PCA(n_components=n_components_95)
# image_pca = pca.fit_transform(image_bw)
#
# image_reconstructed_95 = pca.inverse_transform(image_pca)
#
# plt.imshow(image_reconstructed_95, cmap='gray')
# plt.title('95% Variance')
# plt.grid(False)
# plt.show()


# third part
def encrypt_massage(massage, key_matrix):
    massage_vector = np.array([ord(char) for char in massage])
    eigenvalues_f, eigenvectors_f = np.linalg.eig(key_matrix)
    D_key_matrix = np.dot(np.dot(eigenvectors_f, np.diag(eigenvalues_f)), np.linalg.inv(eigenvectors_f))
    encrypted_vector = np.dot(D_key_matrix, massage_vector)
    return encrypted_vector


def decrypt_massage(encrypted_vector, key_matrix):
    decrypted_massage = ""
    reverse_matrix = np.linalg.inv(key_matrix)
    decrypted_vector = np.dot(reverse_matrix, encrypted_vector)
    for num in decrypted_vector.real:
        decrypted_massage += chr(int(np.round(num)))

    return decrypted_massage


massage = "Hello, word!"
print(f"Original massage: {massage}")

key_matrix = np.random.randint(0, 250, (len(massage), len(massage)))
encrypted_massage = encrypt_massage(massage, key_matrix)
print(f"Encrypted massage:\n{encrypted_massage}")

decrypted_massage = decrypt_massage(encrypted_massage, key_matrix)
print(f"Decrypted massage: {decrypted_massage}")
