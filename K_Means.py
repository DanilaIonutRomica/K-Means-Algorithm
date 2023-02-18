import random
import numpy as np
import sklearn.datasets as skd
import numpy as np
import matplotlib.pyplot as plt

def column(matrix, i):
    return [row[i] for row in matrix]

def calculate_distance(centroid,element,dimension=2):
    sum_of_squares=0
    for i in range(0,dimension):
        sum_of_squares += (centroid[i]-element[i])*(centroid[i]-element[i])
    return np.sqrt(sum_of_squares)

def calculate_centroid(elements,dimension=2):
    sum_dimensions=[]
    for i in range(0,dimension):
        sum_dim_i=0
        for element in elements:
            sum_dim_i+= element[i]
        if len(elements) == 0:
            sum_dimensions.append(0)
        else:
            sum_dim_i/=len(elements)
            sum_dimensions.append(sum_dim_i)
    return sum_dimensions


def K_Means(dataset, k,dimnesion=2):
    dataset_cluster=[0] * len(dataset)
    list_of_previous_centroids = []
    centroids=[]
    #generate random centroids
    
    max_dataset= max(dataset)
    min_dataset= min(dataset)
    max_max=max(max_dataset)
    min_min=min(min_dataset)
    
    for i in range(0,k):
        centroid_dim=[]
        for j in range(0,dimnesion):
            n = random.randint(min_min,max_max)
            centroid_dim.append(n)
        centroids.append(centroid_dim)
    
    counterul=0
    
    for i in range(0,100):
    #while (list_of_previous_centroids != centroids) and (counterul !=5) :
        counterul+=1
        #Calculate distance from element to each centroid
        print("se intampla ceva alooooo")
        distance_matrix=[]
        for centroid in centroids:
            centroid_distances = []
            for element in dataset:
                centroid_distances.append(calculate_distance(centroid,element))
            distance_matrix.append(centroid_distances)

        #Assign each element a centroid
        
        #dataset_cluster=[]
        indexForClusters=0
        for i in range(0,len(dataset)):
            col_i=column(distance_matrix,i)
            #print(col_i)
            index=col_i.index(min(col_i))
            maxm=min(col_i)
            #print(maxm,index)
            for j in range(0,len(centroids)):
                
                if distance_matrix[j][i] == maxm:
                    print(indexForClusters)
                    dataset_cluster[indexForClusters]=centroids[index]
                    indexForClusters+=1
        print(dataset_cluster)
        #Recalculate centroids
        list_of_previous_centroids=centroids
        #print(list_of_previous_centroids)
        li=0
        for centroid in centroids:
            list_of_elements=[]
            i=0
            for element in dataset_cluster:
                if element == centroid:
                    
                    list_of_elements.append(dataset[i])
                i+=1
            centroid=calculate_centroid(list_of_elements)
            centroids[li]=centroid
            li+=1
        #print(centroids)
    #print(centroids,list_of_previous_centroids)    
    
    return [dataset,centroids]
                

            


otherList=[]
for j in range(0,6):
    n = random.randint(1,10000)
    m=random.randint(1,10000)
    randomlist=[m,n]
    otherList.append(randomlist)

#print(otherList)
lista=K_Means(otherList,3)
#print(lista[0][:])
x=[]
y=[]
for elem in lista[0][:]:
    x.append(elem[0])
    y.append(elem[1])

xx=[]
yy=[]
for elem in lista[1][:]:
    xx.append(elem[0])
    yy.append(elem[1])
#print(xx,yy)
plt.plot(x,y, 'o', color='black')
plt.plot(xx,yy,'o', color='red')
plt.show()