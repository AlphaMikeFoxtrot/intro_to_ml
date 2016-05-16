
def sort_by_ones(alist,max_val,a_bin_list,a_bin_len_list):
    """
    alist is a list
    a_bin_list is a dictionary of binary representations of each number in alist
    a_bin_len_list is a dictionary of binary representations of each number in alist
    a_index_list is a dictionary of the current location of each number's index in the final sorted list
    """
    sorted_list = [max_val]
    
    for i in alist:
        a_bin_list[i]
        if a_bin_len_list[i] > a_bin_len_list[sorted_list[-1]]:
            sorted_list.append(i)
        elif a_bin_len_list[i] < a_bin_len_list[sorted_list[-1]]:
            result = dyn_binary_search(i,sorted_list)
            if type(result) != type(tuple()):
                sorted_list.insert(result,i)
            else:
                sorted_list.insert(result[1],i)
                    
        else:
            i_lead_ones = num_leading_ones(a_bin_list[i])
            sorted_lead_ones = num_leading_ones(a_bin_list[sorted_list[-1]])
            if i_lead_ones > sorted_lead_ones:
                sorted_list.append(i)
            elif i_lead_ones < sorted_lead_ones:
                #need to do binary search for correct place for this
                result = dyn_binary_search(i,sorted_list)
                if type(result) != type(tuple()):
                    sorted_list.insert(result,i)
                else:
                    sorted_list.insert(result[1],i)
            else:
                if i >= sorted_list[-1]:
                    sorted_list.append(i)
                else:
                    #need to do binary search for correct place for this
                    result = dyn_binary_search(i,sorted_list)
                    if type(result) != type(tuple()):
                        sorted_list.insert(result,i)
                    else:
                        sorted_list.insert(result[1],i)
    return sorted_list    
        
def num_leading_ones(bin_num):
    bin_num = str(bin_num) #?  I'm not sure if I should/need to do this, but I will for now -
    #bin_num's will probably be strings since they have no reason to be mutable.
    count = 0
    for ind in bin_num[::-1]:
        if ind == '1':
            count += 1
        else:
            return count
    return count


if __name__ == '__main__':
    import random
    from time import time
    a_list = [random.randint(0,100000) for i in range(1000)]
    #a_list = [elem for elem in range(100)]
    #a_list.reverse()
    max_val = max(a_list)
    sorted_list = a_list[:]
    sorted_list.sort()
    a_bin_list = {elem:"{0:b}".format(elem) for elem in a_list}
    a_bin_len_list = {elem:len("{0:b}".format(elem)) for elem in a_list}
    #start = time()
    #listing = sort_by_ones(a_list,max_val,a_bin_list,a_bin_len_list)
    
    #print(listing)
    #print(time() - start)
    start = time()
    print(sort_by_value(a_list,max_val) == sorted_list)
    print(time() - start)
