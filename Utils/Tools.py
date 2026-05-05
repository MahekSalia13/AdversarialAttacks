

def write_data_in_file(filename, data):
    with open(filename, 'a') as the_file:
        ch, attack, eps, rob_acc, test_acc = data
        ch = ','.join(str(x) for x in ch)

        the_file.write(ch + '\n')
        the_file.write(attack + '\n')
        the_file.write(str(eps) + '\n')
        the_file.write(str(rob_acc) + '\n')
        the_file.write(str(test_acc) + '\n')

def string_to_number(lst):
    result =[]
    for s in lst:
        if float(s) % 1 == 0:
            result.append(int(s))
        else:
            result.append(float(s))
    return result

def st_to_number(str):
    if float(str) % 1 == 0:
        return int(str)
    else:
        return float(str)


def list_to_string(lst):
    s = [str(c) for c in lst]
    return ','.join(s)

def read_data_from_file(filename):
    import pandas as pd
    test_data = {'chromosome': [], 'attack': [], 'eps': [], 'rob_acc': [], 'test_acc': []}
    df_data = pd.DataFrame(test_data)
    line_data = open(filename).read().splitlines()
    f = 0
    for lines in line_data:
        if f == 0:
            ch = lines
        elif f == 1:
            attck = lines
        elif f == 2:
            eps = st_to_number(lines)
        elif f == 3:
            rob_acc = float(lines)
        elif f == 4:
            test_acc = float(lines)
            df_data.loc[len(df_data)] = [ch, attck, eps, rob_acc, test_acc]
            f = -1
        f += 1
    return df_data
