def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, _, selected_leaf_idxs = list(zip(*data))

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    selected_leaf_idxs = [torch.tensor(i, dtype=torch.int64) for i in selected_leaf_idxs]

    # Collect frequent noun ids (useful for negative sampling)
    selected_noun_ids = []
    for i, selected_leaf_idxs_per_sample in enumerate(selected_leaf_idxs):
        selected_noun_ids_per_sample = []
        for idx in selected_leaf_idxs_per_sample:
            selected_noun_ids_per_sample.append(captions[i][idx].item())
        selected_noun_ids.append(selected_noun_ids_per_sample)

    return images, targets, lengths, ids, None, selected_leaf_idxs, selected_noun_ids