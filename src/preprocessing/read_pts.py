def read_pts_file(file_path):
    """
    Đọc file .pts (định dạng 68 điểm landmark).
    Trả về danh sách [(x, y), ...].
    """
    points = []
    inside = False
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                inside = True
                continue
            elif line.startswith("}"):
                break
            elif inside and line:
                try:
                    x, y = map(float, line.split())
                    points.append((x, y))
                except ValueError:
                    continue
    return points
