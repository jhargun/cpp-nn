CXX = g++
CXXFLAGS = -std=c++14 -Wall -MMD -g
EXEC = nn.out
OBJECTS = matrix.o mlp.o main.o
DEPENDS = ${OBJECTS:.o=.d}

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}


.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}
