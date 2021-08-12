# import benchmarks.cec as cec
# 导入自己写的函数模块
from my_cec import select_dim

# print(type(cec.cec13),len(cec.cec13))
# print(type(cec.cec17),len(cec.cec17))


class CEC(object):
    def __init__(self, cec_year, func_id, dim):
        self.cec_year = cec_year
        self.func_id = func_id
        self.func_name = 'CEC%s[%d]' % (cec_year, func_id)
        self.cec13_dict = select_dim(dim)   # 调用
        self.cec17_dict = None
        if cec_year == '2013':
            # 调用。id是从1-28，下标是从0-27，cec13是一个字典，value是python函数，输入矩阵返回函数值
            # 可以写一个cec13函数，参数是id0-27，返回一个函数，即id对应的函数
            # self.func = cec13[func_id - 1]
            self.func = self.cec13_dict[func_id-1]

            self.optimal = -100.0 * (15 - func_id) if func_id <= 14 else 100.0 * (func_id - 14)
        elif cec_year == '2017':
            # self.func = cec17_dict[func_id - 1]
            self.optimal = 100.0 * func_id
        else:
            raise Exception('In CEC: Invalid parameter cec_year.')
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0

    def get_dim(self):
        return self.dim

    def get_lb(self):
        return self.lb

    def get_ub(self, dim=0):
        return self.ub

    def get_fitness(self, x):
        """
        get fitness value given x
        :param x: matrix x to be calculated, size = [sample_num * dim]
        :return: fitness values, size = [sample_num]
        """
        return self.func(x)

    def get_func_name(self):
        """to print info"""
        return self.func_name

    def get_func_name_title(self):
        """绘图时的标题"""
        return self.cec13_dict[self.func_id-1].__name__

    def get_optimal(self):
        return self.optimal

    def print_info(self):
        print('CEC_%s[%d]  dim:%d  lb=%d  ub=%d' % (self.cec_year, self.func_id, self.dim, self.lb, self.ub))
