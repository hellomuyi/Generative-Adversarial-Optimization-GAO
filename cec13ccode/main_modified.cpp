/*
  CEC13 Test function suite 
  Jane Jing Liang (email: liangjing@zzu.edu.cn) 
  Dec. 23th 2012
*/

#include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "test_func_modified.cpp"	 

//void test_func(double *, double *,int,int,int);

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;		//test_func.cpp，第52行声明的外部变量 


int main()
{
	int i,j,k,n,m,func_num;
	double *f,*x;
	FILE *fpt;

	m=2;		//sample_num
	n=10;		//dim

		fpt=fopen("input_data/shift_data.txt","r");
		if (fpt==NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		x=(double *)malloc(m*n*sizeof(double));	// x 存放解，逻辑上是m*n矩阵，但存储结构是一维数组 
		if (x==NULL)
			printf("\nError: there is insufficient memory available!\n");
		for(i=0;i<n;i++)						//存放shift，前n个 
		{
				fscanf(fpt,"%Lf",&x[i]);
				printf("%Lf\n",x[i]);		
		}
		fclose(fpt);

		for (i = 1; i < m; i++)					//后(m-1)*n个	
		{
			for (j = 0; j < n; j++)
			{
				x[i*n+j]=0.0;
				printf("%Lf\n",x[i*n+j]);
			}
		}


	f=(double *)malloc(sizeof(double)  *  m);	// f 存放每个sample的函数值，sample_num=m=2 
	for (i = 0; i < 28; i++)
	//for (i = 21; i < 22; i++)
	{
		func_num=i+1;
		for (k = 0; k < 1; k++)	//每个函数的实验次数 
		{
			test_func(x, f, n,m,func_num);		//调用，此时的x前n个是平移值，后(m-1)*n个是0 ，返回m个函数值 
			for (j = 0; j < m; j++)				//打印每个sample的结果 
				printf(" f%d(x[%d]) = %Lf,",func_num,j+1,f[j]);
			printf("\n");
		}
	}
	free(x);
	free(f);
	free(y);
	free(z);
	free(M);
	free(OShift);
	free(x_bound);
	
	system("pause");
	
	return 0;
}


