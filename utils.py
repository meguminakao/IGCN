import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx

def normalize_adj(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -1).flatten()
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return d_mat_inv_sqrt.dot(adj).tocoo()

def preprocess_adj(adj):
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return sparse_to_tuple(adj_normalized)

def construct_feed_dict(features, labels, img_inp, img_label, shapes, ipos, adj, rmax, face, face_norm, support, placeholders):
	feed_dict = dict()
	feed_dict.update({placeholders['features']: features})
	feed_dict.update({placeholders['labels']: labels})
	feed_dict.update({placeholders['img_inp']: img_inp})
	feed_dict.update({placeholders['img_label']: img_label})
	feed_dict.update({placeholders['shapes']: shapes})
	feed_dict.update({placeholders['ipos']: ipos})
	feed_dict.update({placeholders['adj']: adj.toarray()})
	feed_dict.update({placeholders['rmax']: rmax})
	feed_dict.update({placeholders['face']: face})
	feed_dict.update({placeholders['face_norm']: face_norm})
	feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})

	return feed_dict

