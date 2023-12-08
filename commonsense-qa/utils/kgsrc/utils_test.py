


def check_is_sub_dict(sub_dic, super_dic):
    '''check whether a 2d dic is a subset of a 2d super dic'''
    ''' #toy case
    super_dic = {'gfg' : {"a":1, "a1":2},
                    'is' :{"b":2} ,
                    'best' : {"c":3},
                    'for' : {"d:":4}}
    sub_dic = {'gfg' : {"a":1}, 'is' :{"b":2}}

    # printing original dictionaries
    #print("The original dictionary 1 : " +  str(super_dic))
    #print("The original dictionary 2 : " +  str(sub_dic))
    # Using items() + <= operator
    # Check if one dictionary is subset of other
    res = sub_dic['gfg'].items() <= super_dic['gfg'].items()
    print(sub_dic['gfg'].items())
    print(super_dic.items())

    for k in sub_dic.keys():
        assert sub_dic[k].items() <= super_dic[k].items(), ""
        print("Does dict2 lie in dict1 ? : " + str(res))
    # printing result
    '''

    for k in sub_dic.keys():
        assert sub_dic[k].items() <= super_dic[k].items(), \
            "{} is not consistent in sub_dic:{}\n and super_dic:{}".format(k, sub_dic[k], super_dic[k])

#def check_is_sub_list(sub_list, super_list):
