#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <errno.h>

//PROJET PROG DISTRIBUEE : HOWARD

void displayMatrix(int * tab, int nbL, int nbC) {
	for(int i = 0; i < nbL; i++){
		for(int j = 0; j < nbC; j++){
			printf("%i ", tab[i*nbC+j]);
		}
		printf("\n");
	}
}

void parallelCopy(int * tabSrc, int * tabDest, int from, int to){
	#pragma omp parallel for	
	for(int i = from; i < to; i++){
		tabDest[i-from] = tabSrc[i];
	}
}

void parallelInitR(int * matriceResultat, int size){
	#pragma omp parallel for	
	for(int i = 0; i < size; i++){
		matriceResultat[i] = 0;
	}
}

void sumMatrix(int * result, int * toAdd, int size){
	#pragma omp parallel for	
	for(int i = 0; i < size; i++){
		result[i] += toAdd[i];
	}
}

int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	MPI_Status status;
	int * dimensions;
	int * soustabA;
	int * soustabB;
	int * soustabBExpended;
	int * R;
	if(world_rank == 0){
		
		char * nameFileA = argv[1];
		char * nameFileB = argv[2];
		/*
		****PARSING****
		*/
		int * tabA;
		int * tabB;
		int nbEA = 0;
		int nbLA = 0;
		int nbCA = 0;
		int nbEB = 0;
		int nbLB = 0;
		int nbCB = 0;
		char current;
		int currentInt;
		int alreadyReadInt = 0;
		FILE* fichier = NULL;
		fichier = fopen(nameFileA, "r");
		if (fichier != NULL)
		{
			//Dimensions of matrix A
			while((current = fgetc(fichier)) != EOF){
				if(current != ' '){
					if(current == '\n'){
						nbLA++;
						alreadyReadInt=0;
					}else{
						if(alreadyReadInt==0){
							nbEA++;
							alreadyReadInt=1;
						}
					}
				}else{
					alreadyReadInt=0;
				}
			}
			nbCA = nbEA/nbLA;
			//Parsing of matrix A
			rewind(fichier);
			tabA = malloc(nbEA*sizeof(int));
			for(int iiA=0; iiA<nbLA; iiA++){
				for(int jjA=0; jjA<nbCA; jjA++){
					if(fscanf(fichier, "%d", &currentInt)== 1){
						tabA[iiA*nbLA+jjA] = currentInt;
					}
				}
			}
			fclose(fichier);
		}

		fichier = NULL;
		fichier = fopen(nameFileB, "r");
		alreadyReadInt=0;
		if (fichier != NULL)
		{
			//Dimensions of matrix B
			while((current = fgetc(fichier)) != EOF){
				if(current != ' '){
					if(current == '\n'){
						nbLB++;
						alreadyReadInt=0;
					}else{
						if(alreadyReadInt==0){
							nbEB++;
							alreadyReadInt=1;
						}
					}
				}else{
					alreadyReadInt=0;
				}
			}
			nbCB = nbEB/nbLB;
			//Parsing of matrix B + invert ligne and column for easier read
			rewind(fichier);
			tabB = malloc(nbEB*sizeof(int));
			for(int iiB=0; iiB<nbLB; iiB++){
				for(int jjB=0; jjB<nbCB; jjB++){
					if(fscanf(fichier, "%d", &currentInt)== 1){
						tabB[jjB*nbCB+iiB] = currentInt;
					}
				}
			}
			fclose(fichier);
		}

		/*
		****BROADCAST DIMENSIONS OF BOTH MATRIX****
		*/
		dimensions = malloc(6*sizeof(int));
		dimensions[0]=nbEA;
		dimensions[1]=nbLA;
		dimensions[2]=nbCA;
		dimensions[3]=nbEB;
		dimensions[4]=nbLB;
		dimensions[5]=nbCB;
		MPI_Send(dimensions, 6, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
		/*
		****SCATTER A [0->P-2]****
		*/
		int tailleSousTabA = (nbLA/world_size)*nbCA;
		soustabA = malloc(tailleSousTabA*sizeof(int));
		for(int scatA = world_size-2; scatA >=0 ; scatA--){
			parallelCopy(tabA, soustabA, scatA*tailleSousTabA, ((scatA+1)*tailleSousTabA));
			MPI_Send(soustabA, tailleSousTabA, MPI_INT, (world_rank+1)%world_size, 1, MPI_COMM_WORLD);
		}

		/*
		****KEEP LAST PART OF A [P-1->END]****
		*/
		int tailleSousTabAExpended = ((nbLA/world_size) + (nbLA%world_size)) * nbCA;
		int * soustabAExpended = malloc(tailleSousTabAExpended*sizeof(int));
		parallelCopy(tabA, soustabAExpended, (world_size-1)*tailleSousTabA, nbEA);

		/*
		****CALCULE A[P-1:] x B[0->P-2] AND DISTRIBUTION OF B[0->P-2]****
		*/
		R = malloc(nbLA*nbCB*sizeof(int));
		parallelInitR(R, nbLA*nbCB);
		int tailleSousTabB = (nbCB/world_size)*nbLB;
		soustabB = malloc(tailleSousTabB*sizeof(int));
		int nbLAbis = (nbLA/world_size) + (nbLA%world_size);
		int nbCBbis = (nbCB/world_size);
		int startA = world_size-1;
		//Une ligne de sousA par processeur à multiplier avec les colonnes de B
		for(int scatB = 0; scatB < world_size-1; scatB++){
			//sousTabB à envoyer et traiter
			parallelCopy(tabB, soustabB, scatB*tailleSousTabB, ((scatB+1)*tailleSousTabB));
			MPI_Send(soustabB, tailleSousTabB, MPI_INT, (world_rank+1)%world_size, 2, MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int iA = 0; iA < nbLAbis; iA++){
				int prodToSum;
				for(int jB = 0; jB < nbCBbis; jB++){
					prodToSum=0;
					for(int k = 0; k < nbCA; k++){
						prodToSum = prodToSum + (soustabAExpended[k+(iA*nbCA)] * soustabB[k+(jB*nbLB)]);
					}
					R[((startA+iA)*nbCB)+(scatB+jB)] = prodToSum; //BP
				}
			}
		}


		/*
		****LAST B [P-1->END]****
		*/
		int tailleSousTabBExpended = ((nbCB/world_size) + (nbCB%world_size)) * nbLB;
		soustabBExpended = malloc(tailleSousTabBExpended*sizeof(int));
		parallelCopy(tabB, soustabBExpended, (world_size-1)*tailleSousTabB, nbEB);
		nbCBbis = (nbCB/world_size) + (nbCB%world_size);
		MPI_Send(soustabBExpended, tailleSousTabBExpended, MPI_INT, (world_rank+1)%world_size, 3, MPI_COMM_WORLD);
		#pragma omp parallel for
		for(int iAb = 0; iAb < nbLAbis; iAb++){
			int prodToSum;
			for(int jBb = 0; jBb < nbCBbis; jBb++){
				prodToSum=0;
				for(int k = 0; k < nbCA; k++){
					prodToSum = prodToSum + (soustabAExpended[k+(iAb*nbCA)] * soustabBExpended[k+(jBb*nbLB)]);
				}
				R[((startA+iAb)*nbCB)+((world_size-1)+jBb)] = prodToSum; //BP
			}
		}

		/*
		****RECEIVING ACCUMULATED PARTIAL RESULTS AND ADDING IT TO OUR CURRENT PARTIAL RESULT / GATHER****
		*/
		int * Rtmp = malloc(nbLA*nbCB*sizeof(int));
		int tailleR = nbLA*nbCB;
		MPI_Recv(Rtmp, tailleR, MPI_INT, (world_rank+1)%world_size, 4, MPI_COMM_WORLD, &status);
		sumMatrix(R,Rtmp,tailleR);

		/*
		****DISPLAY****
		*/
		displayMatrix(R,nbLA,nbCB);

	}else if(world_rank == world_size-1){ //le last
		/*
		****GET DIMENSIONS OF BOTH MATRIX****
		*/
		int nbLA = 0;
		int nbCA = 0;
		int nbLB = 0;
		int nbCB = 0;
		dimensions = malloc(6*sizeof(int));
		MPI_Recv(dimensions, 6, MPI_INT, (world_rank-1+world_size)%world_size, 0, MPI_COMM_WORLD, &status);
		nbLA = dimensions[1];
		nbCA = dimensions[2];
		nbLB = dimensions[4];
		nbCB = dimensions[5];

		/*
		****RECEIVE HIS PART OF A****
		*/
		int tailleSousTabA = (nbLA/world_size)*nbCA;
		soustabA = malloc(tailleSousTabA*sizeof(int));
		MPI_Recv(soustabA, tailleSousTabA, MPI_INT, (world_rank-1+world_size)%world_size, 1, MPI_COMM_WORLD, &status);

		/*
		****CALCULE A[P-2:] x B[0->P-2]****
		*/
		R = malloc(nbLA*nbCB*sizeof(int));
		parallelInitR(R, nbLA*nbCB);
		int tailleSousTabB = (nbCB/world_size)*nbLB;
		soustabB = malloc(tailleSousTabB*sizeof(int));
		int nbLAbis = (nbLA/world_size);
		int nbCBbis = (nbCB/world_size);
		int startA = (world_rank-1+world_size)%world_size;
		//Une ligne de sousA par processeur à multiplier avec les colonnes de B
		for(int scatB = 0; scatB < world_size-1; scatB++){
			MPI_Recv(soustabB, tailleSousTabB, MPI_INT, (world_rank-1+world_size)%world_size, 2, MPI_COMM_WORLD, &status);
			#pragma omp parallel for
			for(int iA = 0; iA < nbLAbis; iA++){
				int prodToSum;
				for(int jB = 0; jB < nbCBbis; jB++){
					prodToSum=0;
					for(int k = 0; k < nbCA; k++){
						prodToSum = prodToSum + (soustabA[k+(iA*nbCA)] * soustabB[k+(jB*nbLB)]);
					}
					R[((startA+iA)*nbCB)+(scatB+jB)] = prodToSum; //BP
				}
			}
		}

		/*
		****LAST B [P-1->END]****
		*/
		int tailleSousTabBExpended = ((nbCB/world_size) + (nbCB%world_size)) * nbLB;
		soustabBExpended = malloc(tailleSousTabBExpended*sizeof(int));
		nbCBbis = (nbCB/world_size) + (nbCB%world_size);
		MPI_Recv(soustabBExpended, tailleSousTabBExpended, MPI_INT, (world_rank-1+world_size)%world_size, 3, MPI_COMM_WORLD, &status);
		#pragma omp parallel for
		for(int iAb = 0; iAb < nbLAbis; iAb++){
			int prodToSum;
			for(int jBb = 0; jBb < nbCBbis; jBb++){
				prodToSum=0;
				for(int k = 0; k < nbCA; k++){
					prodToSum = prodToSum + (soustabA[k+(iAb*nbCA)] * soustabBExpended[k+(jBb*nbLB)]);
				}
				R[((startA+iAb)*nbCB)+((world_size-1)+jBb)] = prodToSum; //BP
			}
		}

		/*
		****SEND PARTIAL RESULTS / GATHER****
		*/
		int tailleR = nbLA*nbCB;
		MPI_Send(R, tailleR, MPI_INT, (world_rank-1+world_size)%world_size, 4, MPI_COMM_WORLD);

	}else{ //les autres
		
		/*
		****GET DIMENSIONS OF BOTH MATRIX****
		*/
		int nbLA = 0;
		int nbCA = 0;
		int nbLB = 0;
		int nbCB = 0;
		dimensions = malloc(6*sizeof(int));
		MPI_Recv(dimensions, 6, MPI_INT, (world_rank-1+world_size)%world_size, 0, MPI_COMM_WORLD, &status);
		MPI_Send(dimensions, 6, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
		nbLA = dimensions[1];
		nbCA = dimensions[2];
		nbLB = dimensions[4];
		nbCB = dimensions[5];

		/*
		****RECEIVE HIS PART OF A WHILE RETRANSMITTING THE OTHERS PARTS****
		*/
		int tailleSousTabA = (nbLA/world_size)*nbCA;
		soustabA = malloc(tailleSousTabA*sizeof(int));
		MPI_Recv(soustabA, tailleSousTabA, MPI_INT, (world_rank-1+world_size)%world_size, 1, MPI_COMM_WORLD, &status);
		for(int skip = 0; skip < (world_size-1-world_rank)%world_size; skip++){
			MPI_Send(soustabA, tailleSousTabA, MPI_INT, (world_rank+1)%world_size, 1, MPI_COMM_WORLD);
			MPI_Recv(soustabA, tailleSousTabA, MPI_INT, (world_rank-1+world_size)%world_size, 1, MPI_COMM_WORLD, &status);
		}

		/*
		****CALCULE A[rank-1] x B[0->P-2]****
		*/
		R = malloc(nbLA*nbCB*sizeof(int));
		parallelInitR(R, nbLA*nbCB);
		int tailleSousTabB = (nbCB/world_size)*nbLB;
		soustabB = malloc(tailleSousTabB*sizeof(int));
		int nbLAbis = (nbLA/world_size);
		int nbCBbis = (nbCB/world_size);
		int startA = (world_rank-1+world_size)%world_size;
		//Une ligne de sousA par processeur à multiplier avec les colonnes de B
		for(int scatB = 0; scatB < world_size-1; scatB++){
			MPI_Recv(soustabB, tailleSousTabB, MPI_INT, (world_rank-1+world_size)%world_size, 2, MPI_COMM_WORLD, &status);
			MPI_Send(soustabB, tailleSousTabB, MPI_INT, (world_rank+1)%world_size, 2, MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int iA = 0; iA < nbLAbis; iA++){
				int prodToSum;
				for(int jB = 0; jB < nbCBbis; jB++){
					prodToSum=0;
					for(int k = 0; k < nbCA; k++){
						prodToSum = prodToSum + (soustabA[k+(iA*nbCA)] * soustabB[k+(jB*nbLB)]);
					}
					R[((startA+iA)*nbCB)+(scatB+jB)] = prodToSum; //BP
				}
			}
		}

		/*
		****LAST B [P-1->END]****
		*/
		int tailleSousTabBExpended = ((nbCB/world_size) + (nbCB%world_size)) * nbLB;
		soustabBExpended = malloc(tailleSousTabBExpended*sizeof(int));
		nbCBbis = (nbCB/world_size) + (nbCB%world_size);
		MPI_Recv(soustabBExpended, tailleSousTabBExpended, MPI_INT, (world_rank-1+world_size)%world_size, 3, MPI_COMM_WORLD, &status);
		MPI_Send(soustabBExpended, tailleSousTabBExpended, MPI_INT, (world_rank+1)%world_size, 3, MPI_COMM_WORLD);
		#pragma omp parallel for
		for(int iAb = 0; iAb < nbLAbis; iAb++){
			int prodToSum;
			for(int jBb = 0; jBb < nbCBbis; jBb++){
				prodToSum=0;
				for(int k = 0; k < nbCA; k++){
					prodToSum = prodToSum + (soustabA[k+(iAb*nbCA)] * soustabBExpended[k+(jBb*nbLB)]);
				}
				R[((startA+iAb)*nbCB)+((world_size-1)+jBb)] = prodToSum; //BP
			}
		}

		/*
		****RECEIVING ACCUMULATED PARTIAL RESULTS AND ADDING IT TO OUR CURRENT PARTIAL RESULT / GATHER****
		*/
		int * Rtmp = malloc(nbLA*nbCB*sizeof(int));
		int tailleR = nbLA*nbCB;
		MPI_Recv(Rtmp, tailleR, MPI_INT, (world_rank+1)%world_size, 4, MPI_COMM_WORLD, &status);
		sumMatrix(R,Rtmp,tailleR);

		/*
		****SEND PARTIAL RESULTS / GATHER****
		*/
		MPI_Send(R, tailleR, MPI_INT, (world_rank-1+world_size)%world_size, 4, MPI_COMM_WORLD);
	}
	// Finalize the MPI environment. 
	free(dimensions);
	free(soustabA);
	free(soustabB);
	free(soustabBExpended);
	free(R);
	MPI_Finalize();

}

