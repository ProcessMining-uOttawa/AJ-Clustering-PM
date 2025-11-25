from enum import Enum
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.visualization.heuristics_net.variants.pydotplus_vis import get_graph as hn_get_graph
from pm4py.statistics.attributes.log import get as attr_get
import pandas as pd

def get_log(path):
    log = pd.read_csv(path, index_col=False)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    return log

class ProcAnn(Enum):
    FREQ = "frequency"
    FREQ_PERC = "frequency (percentage)"
    PERF = "performance"

def mine_vis(visualizer, gviz, output_path, save_gviz=False):
    if output_path is not None:
        visualizer.save(gviz, f"{output_path}.jpg")
        if save_gviz:
            gviz.save(f"{output_path}.gv")
    else:
        visualizer.view(gviz)

def mine_dfg(log, ann=ProcAnn.FREQ, output_path=None, save_gviz=False):
    match ann:
        case ProcAnn.FREQ | ProcAnn.FREQ_PERC:
            mine_var = dfg_discovery.Variants.FREQUENCY
            vis_var = dfg_visualizer.Variants.FREQUENCY
        case ProcAnn.PERF:
            mine_var = dfg_discovery.Variants.PERFORMANCE
            vis_var = dfg_visualizer.Variants.PERFORMANCE
    
    # discover
    parameters = { 'pm4py:param:start_timestamp_key': 'time:timestamp' }
    dfg = dfg_discovery.apply(log, variant=mine_var, parameters = parameters)

    activ_count = None
    if ann == ProcAnn.FREQ_PERC:
        num_traces = log['case:concept:name'].nunique()
        dfg = { key: round((count / num_traces), 2) for key, count in dfg.items() }

        activ_count = attr_get.get_attribute_values(log, 'concept:name')
        activ_count = { key: round((count / num_traces), 2) for key, count in activ_count.items() }

    # visualize
    gviz = dfg_visualizer.apply(dfg, log=log, variant=vis_var, activities_count=activ_count)
    mine_vis(dfg_visualizer, gviz, output_path, save_gviz)
        

def mine_alpha(log, output_path=None, save_gviz=False):
    # alpha miner
    net, initial_marking, final_marking = alpha_miner.apply(log)

    # visualise
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    mine_vis(pn_visualizer, gviz, output_path, save_gviz)
    

def mine_heur(log, ann=ProcAnn.FREQ, output_path=None, save_gviz=False):
    # heuristics miner
    heu_net = heuristics_miner.apply_heu(log, { "heu_net_decoration": ann.value })

    # visualize
    # (works differently ...)
    if save_gviz and output_path is not None:
        gviz = hn_get_graph(heu_net)
        gviz.write(f"{output_path}.gv")
    
    gviz = hn_visualizer.apply(heu_net)
    mine_vis(hn_visualizer, gviz, output_path, False)
    
    
def mine_induct(log, convert_to=None, ann=None, output_path=None, save_gviz=False):
    # create the process tree
    # (wvw: drop "_tree" from call)
    tree = inductive_miner.apply(log)

    if convert_to is not None:
        match (convert_to):
            case 'petri_net': 
                net, initial_marking, final_marking = process_tree_converter.apply(tree)
                variant = pn_visualizer.WO_DECORATION
                if ann is not None:
                    match (ann):
                        case ProcAnn.FREQ:
                            variant = pn_visualizer.FREQUENCY_DECORATION
                        case ProcAnn.PERF:
                            variant = pn_visualizer.PERFORMANCE_DECORATION
                gviz = pn_visualizer.apply(net, initial_marking, final_marking, log=log, variant=variant)
            case 'bpmn':
                bpmn = process_tree_converter.apply(tree, variant=process_tree_converter.Variants.TO_BPMN)
                gviz = bpmn_visualizer.apply(bpmn)
            case _:
                raise f"Unsupported format: {convert_to}"
        
        mine_vis(pn_visualizer, gviz, output_path, save_gviz)
    else:
        gviz = pt_visualizer.apply(tree)
        mine_vis(pt_visualizer, gviz, output_path, save_gviz)
        
        
def mine_ilp(log, output_path=None, save_gviz=False):
    # heuristics miner
    net, init_mark, final_mark = ilp_miner.apply(log)

    # visualize
    gviz = pn_visualizer.apply(net, init_mark, final_mark)
    mine_vis(pn_visualizer, gviz, output_path, save_gviz)
    
    
def log_subset_horizontal(log, perc):
    new_len = int(log.shape[0] * perc)
    new_log = log.iloc[0:new_len]
    
    # (ensures that full cases are extracted)
    cutoff_case = log.iloc[new_len-1]['case:concept:name']
    copied_len = new_log[new_log['case:concept:name']==cutoff_case].shape[0]
    total_len = log[log['case:concept:name']==cutoff_case].shape[0]

    if copied_len != total_len:
        new_len = new_len + (total_len - copied_len)
        new_log = log.iloc[0:new_len]

    return new_log


def log_subset_vertical(log, perc):
    print("original:")
    counts = log.groupby('case:concept:name')['concept:name'].count()
    print(counts.describe())
    
    case_logs = [ df for _, df in log.groupby('case:concept:name') ]
    case_logs = map(lambda df: df.iloc[0:int(df.shape[0]*perc)], case_logs)
    
    log_subset = pd.concat(case_logs)
    
    print("\nsubset:")
    counts = log_subset.groupby('case:concept:name')['concept:name'].count()
    print(counts.describe())
    
    return log_subset


def get_time_diff(log):
    log = log.drop([ 'index', 'time_diff' ], axis=1, errors='ignore') # drop any prior columns (if any)
    log = log.sort_values(by=['case:concept:name', 'time:timestamp', 'concept:name'])
    log = log.reset_index().drop('index', axis=1) # reset & forget current index
    log = log.reset_index() # get current index as column
    
    cases_firsts = log.groupby('case:concept:name').first().reset_index()
    cases_firsts
    
    log['time_diff'] = log['time:timestamp'] - log['time:timestamp'].shift(1)
    log['time_diff'] = log['time_diff'].apply(pd.Timedelta.total_seconds)
    log['time_diff']
    
    # irrelevant time differences for first events of cases
    log.loc[cases_firsts['index'], 'time_diff'] = 1_000_000
    # print(log[log['time_diff']==1_000_000].shape[0]) # sanity check
    
    return log


def equal_timestamps_interval(log, interval):
    if 'time_diff' not in log.columns:
        log = get_time_diff(log)
    
    # set time_diff2 to 0 if time_diff is less than <interval>
    log['time_diff2'] = log['time_diff']
    log.loc[log['time_diff2'] < interval, 'time_diff2'] = 0
    
    # get all rows where time difference was _not_ less than interval
    intervals = log[log['time_diff2'] > 0][['index', 'time:timestamp']]
    # these rows' timestamps will apply to all subsequent rows where time_diff2 == 0
    # (so, until the next row in this df)
    intervals['from'] = intervals['index']
    intervals['to'] = intervals['index'].shift(-1)
    # set last "to" to max value of index
    intervals.iloc[-1, 3] = log['index'].max() + 1 # (to get an inclusive interval of max)
    # print(intervals)
    
    # convert interval into lists with all relevant indexes
    # (later, we can "explode" these into their own rows)

    def gen_interval(row):
        if row.iloc[0] == row.iloc[1]:
            return [int(row.iloc[0])]
        else:
            return range(int(row.iloc[0]), int(row.iloc[1]))

    intervals['list'] = intervals[['from', 'to']].apply(gen_interval, axis=1)
    # use timestamp as index; we need them in the explode result
    # (we know they're unique at this point)
    intervals = intervals.set_index(intervals['time:timestamp'])
    intervals = intervals[['list']]
    # print(intervals)
    
    # explode the interval lists into separate rows

    # add timestamp index as separate column (reset index)
    intervals = intervals['list'].explode().reset_index()
    intervals = intervals.rename({'time:timestamp': 'time:timestamp2', 'list': 'index'}, axis=1)
    # print(intervals)
    
    # join log with intervals based on the index 
    # for now, their new timestamp will be "time:timestamp2"
    # (use left join; rows with time_diff2 > 0 will be N/A)
    log2 = log.merge(intervals, left_on='index', right_on='index', how='left')
    # for cases where time_diff2 > 0, simply copy the original timestamps
    log2.loc[log2['time_diff2']>0, 'time:timestamp2'] = log2.loc[log2['time_diff2']>0, 'time:timestamp']
    
    # drop current timestamp & unused columns (but keep index)
    log2 = log2.drop([ 'time_diff', 'time_diff2', 'time:timestamp' ], axis=1)
    # this is our new timestamp
    log2 = log2.rename({ 'time:timestamp2': 'time:timestamp' }, axis=1)
    
    log2 = log2.sort_values(by=['case:concept:name', 'time:timestamp', 'concept:name'])
    return log2

    # alternative loop-based solution
    # # (takes 3-4 minutes vs. 2-4 seconds)

    # log['time:timestamp2'] = log['time:timestamp']

    # cur_ts = None
    # for idx, row in log.iterrows():
    #     if (idx % 1000 == 0):
    #         print(idx, round((idx / log.shape[0]) * 100, 2), "%")
        
    #     if row['time_diff2'] > 0: # always the case for the first row
    #          cur_ts = row['time:timestamp']
        
    #     else:
    #         log.loc[idx, 'time:timestamp2'] = cur_ts


def aggregate_events(log, events, max_timedelta, repl=None, verbose=False):
    # log = log.copy()
    
    # - uses loops but more robust
    
    total_groups = 0; total_size = 0
    total_simult = 0; total_diff = 0; non_simul = []
    # activ_orders = {
    #     (events[0], events[-1]): 0, 
    #     (events[-1], events[0]): 0, 
    # }
    
    class Group:        
        def __init__(self):
            self.events = []
            self.idxes = []
        def add(self, event):
            self.events.append(event['concept:name'])
            self.idxes.append(event.name)
        def includes(self, evt_name):
            return evt_name in self.events
        def indexes(self):
            return self.idxes
        def size(self):
            return len(self.events)
        def __str__(self):
            return " (" + ", ".join([ f"'{evt}'@{idx}" for evt, idx in zip(self.events, self.idxes) ]) + ") "
    
    class Case:        
        def __init__(self, id=None, verbose=False):
            self.id = id
            self.verbose = verbose;
            if verbose: # only keep groups if verbose
                self.groups = []
        def add(self, group):
            # pass
            if self.verbose:
                self.groups.append(group)
        def size(self):
            if self.verbose:
                return len(self.groups)
            else:
                return 0
        def __str__(self):
            return f"case {self.id}: " + "; ".join(map(lambda g: g.__str__(), self.groups))
    
    to_drop = []
    cur_case = Case(verbose=verbose); cur_group = Group()
    
    def record_case(cur_case, cur_group):
        # deal with leftover (incomplete) group
        cur_group = record_group(cur_case, cur_group) if cur_group.size() > 0 else Group()
        # (testing) print non-empty cases
        if verbose: # and cur_case.size() > 0:
            print(cur_case)
        cur_case = Case(verbose=verbose)
        return cur_case, cur_group
            
    def record_group(cur_case, cur_group, init_evt=None):
        nonlocal total_groups, total_size, total_simult, total_diff #, activ_orders
        if verbose and cur_group.size() != len(events):
            print(f"case {cur_case.id}: non-complete group {cur_group}")
        
        # - replace last event in group with 'repl' event
        # log.loc[cur_group.idxes[-1], 'concept:name'] = repl
        # to_drop.extend(cur_group.idxes[:-1])
        
        # - get timestamp differences using diff()
        # (time differences can be cumulative this way)
        # diff = log.loc[cur_group.idxes]['time:timestamp'].diff().rename('diff').astype(int)
        # simult = diff[(diff >= 0) & (diff <= max_timedelta)]
        # if (len(simult) == cur_group.size()-1):
            # total_simult += 1
        
        # - drop groups with "simultaneous" events (as per max_timedelta)
        # get timestamp differences between first, last index
        ts1 = log.loc[cur_group.idxes[0], 'time:timestamp']
        ts2 = log.loc[cur_group.idxes[-1], 'time:timestamp']
        # name1 = log.loc[cur_group.idxes[0], 'concept:name']
        # name2 = log.loc[cur_group.idxes[-1], 'concept:name']
        diff = (ts2 - ts1).total_seconds()
        # print(total_diff)
        if diff <= max_timedelta:
            total_simult += 1
            to_drop.extend(cur_group.idxes)
        else:
            total_diff += diff
            
        # else:
        #     if ts1 < ts2:
        #         activ_orders[(name1, name2)] += 1
        #     elif ts2 < ts1:
        #         activ_orders[(name2, name1)] += 1
        
        # enforce ordering
        # first = cur_group.idxes[0 if name1 == events[0] else -1]
        # second = cur_group.idxes[-1 if name1 == events[0] else 0]
        # log.loc[second, 'time:timestamp'] = log.loc[first, 'time:timestamp'] + pd.Timedelta(seconds=1)
        
        # housekeeping
        total_groups += 1
        total_size += cur_group.size()
        cur_case.add(cur_group)
        cur_group = Group()
        if init_evt is not None:
            cur_group.add(init_evt)
        return cur_group
    
    cnt = 0; size = log.shape[0]; cur_perc = 0
    def progress():
        nonlocal cnt, size, cur_perc
        perc = int(cnt / size * 100)
        if perc != cur_perc and perc % 5 == 0:
            print(f"{perc}% done")
        cnt += 1
        cur_perc = perc
    
    for _, row in log.iterrows():
        # if verbose:
        progress()
        
        case_id = row['case:concept:name']
        # if new case is found, record prior case
        if cur_case.id is None or cur_case.id != case_id:
            if cur_case.id is not None:
                cur_case, cur_group = record_case(cur_case, cur_group)
                # break # (testing)

            cur_case.id = case_id
        
        cur_evt = row['concept:name']
        if cur_evt in events:
            # current event is already in group;
            # record prior group & start new one (incl. current event)
            if cur_group.includes(cur_evt):
                cur_group = record_group(cur_case, cur_group, row)
            else:
                # add event to current group
                cur_group.add(row)
                # if group is full, record it & start new one
                if cur_group.size() == len(events):
                    cur_group = record_group(cur_case, cur_group)
        
    # record last case
    record_case(cur_case, cur_group)
    
    print("# groups:", total_groups, "total size:", total_size, "avg size:", round(total_size/total_groups, 2))
    print("# groups:", total_groups, "# simult:", total_simult, "avg:", round(total_simult/total_groups*100, 2), "%")
    print("# groups:", total_groups, "avg_diff:", int(total_diff/total_groups), "s", "(", int(total_diff/total_groups/60), "m", ")")
    # print(activ_orders)
    
    return log.drop(to_drop)
    
    # - uses dataframes but makes assumptions
    
    # # put 1 in case activity corresponds to event
    # log['evt_cnt'] = 0
    # log.loc[log['concept:name'].isin(events), 'evt_cnt'] = 1
    
    # # take cumulative sum per case; put in cumul_cnt column
    # # (takes sum of current and all prior evt_cnt values; resets per group)
    # cumul_cnts = log.groupby('case:concept:name')['evt_cnt'].cumsum().rename('cumul_cnt')
    
    # # all rows where activity is in events, and cumul_cnt % len(events) is 0, is the last of the aggregation group
    
    # # replace activity with repl
    # log.loc[(log['concept:name'].isin(events)) & (cumul_cnts % len(events) == 0), 'concept:name'] = repl
    # # drop other events not meeting those criteria (repl may also be in events)
    # # return log
    # return log.loc[(~ log['concept:name'].isin(events)) | (log['concept:name'] == repl), log.columns != 'evt_cnt']
    
    
# stats for events

# count total number of occurrences of events (absolute & percentage)
def count_events(log, evt_col='concept:name', case_col='case:concept:name', plot=True):
    total_evts = len(log[evt_col])
    evt_counts = log.groupby(evt_col)[case_col].count().rename('cnt').to_frame()
    evt_counts['perc'] = 100 / total_evts * evt_counts['cnt']
    evt_counts = evt_counts.sort_values(by='perc', ascending=False)
    # for idx, row in evt_counts.items():
    #     print(idx, row[0], row[1])
    if plot:
        ax = evt_counts[['perc']].plot.bar()
        # ax.xaxis.set_visible(False)
    return evt_counts

# per event, count number of cases that it occurs in (absolute & percentage)
def count_cases_per_event(log, evt_col='concept:name', case_col='case:concept:name', plot=True):
    total_cases = len(log[case_col].unique())
    evts_cases = log.groupby(evt_col)[case_col].unique().rename('cases').to_frame()
    evts_cases['cases'] = evts_cases['cases'].apply(len)
    evts_cases['perc'] = 100 / total_cases * evts_cases['cases']
    evt_cases = evts_cases.sort_values(by='perc', ascending=False)
    if plot:
        ax = evt_cases[['perc']].plot.bar()
        # ax.xaxis.set_visible(False)
    return evt_cases

# filter log
def filter_events_on_counts(log, counts, leq_perc, evt_col='concept:name'):
    keep_evts = counts[counts['perc']>leq_perc].index.array
    return log[log[evt_col].isin(keep_evts)]


# stats for traces

def get_trace_lengths(log, evt_col='concept:name', case_col='case:concept:name', plot=True):
    trace_lens = log.groupby(case_col)[evt_col].count()
    trace_lens = trace_lens.sort_values(ascending=False)
    # for idx, cnt in trace_lens.items():
    #     print(idx, cnt)
    if plot:
        ax = trace_lens.plot.bar()
        ax.xaxis.set_visible(False)
    return trace_lens