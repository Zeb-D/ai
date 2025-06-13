def find_min(arr):
    min_var = arr[0]
    min_idx = 0
    for i in range(1, len(arr)):
        if arr[i] < min_var:
            min_var = arr[i]
            min_idx = i
    return min_idx


def select_sort(arr):
    ret = []
    for i in range(len(arr)):
        min_idx = find_min(arr)
        ret.append(arr.pop(min_idx))
    return ret


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 选择排序
    print("selection sort")
    print(select_sort([2, 3, 54, 79, 0]))
