from sympy import symbols, Or, Not, Implies, satisfiable
Rain = symbols('Rain')
Harry_Visited_Hagrid = symbols('Harry_Visited_Hagrid')
Harry_Visited_Dumbledore = symbols('Harry_Visited_Dumbledore')
sentence_1 = Implies(Not(Rain), Harry_Visited_Hagrid)
sentence_2 = (Or(Harry_Visited_Hagrid, Harry_Visited_Dumbledore) & Not(Harry_Visited_Hagrid & Harry_Visited_Dumbledore))
sentence_3 = Harry_Visited_Dumbledore
knowledge_base = sentence_1 & sentence_2 & sentence_3
solution = satisfiable(knowledge_base,all_models=True)
for model in solution:
    if model[Rain]:
        print("It rained today.")
    else:
        print("There is no rain today.")
