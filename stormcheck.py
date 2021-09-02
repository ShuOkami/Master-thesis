import stormpy
import stormpy.core
import sys
import time


def modelCheck():

    prism_program = stormpy.parse_prism_program("PrismFile/" + fileName + ".pm")

    formula_str = "Rmin=?[LRA]"
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)
    print("Model build completed")
    start = time.time()
    result = stormpy.model_checking(model, properties[0])
    print(result)
    end = time.time()
    print("Storm checker computation time : ", round(end-start, 3), " seconds")
    assert result.result_for_all_states

    initial_state = model.initial_states[0]
    print(result.at(initial_state))


if __name__ == '__main__':
    fileName = sys.argv[1]
    modelCheck()