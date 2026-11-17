def calculate_overlaps(a_list, b_list, bounds=None):
    """bbox IoU (Rectangle 쌍별)"""
    out = []
    for a, b in zip(a_list, b_list):
        ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        inter = ix * iy
        u = a.w * a.h + b.w * b.h - inter
        out.append(inter / u if u > 0 else 0.0)
    return out
