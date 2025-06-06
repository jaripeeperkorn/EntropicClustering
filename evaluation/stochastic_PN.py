
#! all old code: need to be updated



'''
from pm4py.algo.conformance.tokenreplay.variants import token_replay
from pm4py.statistics.variants.log import get as variants_module
from pm4py.objects.petri.petrinet import PetriNet
from pm4py.objects.random_variables.random_variable import RandomVariable
from pm4py.objects.petri import performance_map
from pm4py.simulation.montecarlo.parameters import Parameters
from pm4py.util import exec_utils, constants, xes_constants
from pm4py.algo.conformance.tokenreplay import algorithm as executor





def get_map_from_log_and_net(log, net, initial_marking, final_marking, force_distribution=None, parameters=None):
    stochastic_map = {}

    if parameters is None:
        parameters = {}

    token_replay_variant = exec_utils.get_param_value(Parameters.TOKEN_REPLAY_VARIANT, parameters,
                                                      executor.Variants.TOKEN_REPLAY)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)

    parameters_variants = {constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_key}
    variants_idx = variants_module.get_variants_from_log_trace_idx(log, parameters=parameters_variants)
    variants = variants_module.convert_variants_trace_idx_to_trace_obj(log, variants_idx)

    parameters_tr = {token_replay.Parameters.ACTIVITY_KEY: activity_key, token_replay.Parameters.VARIANTS: variants}

    # do the replay
    aligned_traces = executor.apply(log, net, initial_marking, final_marking, variant=token_replay_variant,
                                        parameters=parameters_tr)

    element_statistics = performance_map.single_element_statistics(log, net, initial_marking,
                                                                   aligned_traces, variants_idx,
                                                                   activity_key=activity_key,
                                                                   timestamp_key=timestamp_key,
                                                                   parameters={"business_hours": False})

    for el in element_statistics:
        if type(el) is PetriNet.Transition and "performance" in element_statistics[el]:
            values = element_statistics[el]["performance"]

            rand = RandomVariable()
            rand.calculate_parameters(values, force_distribution=force_distribution)

            no_of_times_enabled = element_statistics[el]['no_of_times_enabled']
            no_of_times_activated = element_statistics[el]['no_of_times_activated']

            if no_of_times_enabled > 0:
                rand.set_weight(float(no_of_times_activated) / float(no_of_times_enabled))
            else:
                rand.set_weight(0.0)

            stochastic_map[el] = rand

    return stochastic_map
'''