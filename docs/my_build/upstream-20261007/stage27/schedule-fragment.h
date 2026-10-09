struct ScheduleEvent {
    int pos, width, event, raw_first, raw_last, raw_source, physical_first, physical_last;
    std::vector<llama_token> tokens;
};
static std::vector<ScheduleEvent> read_schedule(const std::string & path) {
    std::ifstream input(path);check(bool(input),"schedule open failed");
    std::vector<ScheduleEvent> events;
    for (std::string line;std::getline(input,line);) {
        std::istringstream row(line);ScheduleEvent e;
        check(bool(row>>e.pos>>e.width>>e.event>>e.raw_first>>e.raw_last>>e.raw_source>>e.physical_first>>e.physical_last),"schedule metadata failed");
        check(e.width>0 && e.width<=4,"schedule width failed");e.tokens.resize(e.width);
        for (auto & token:e.tokens)check(bool(row>>token),"schedule tokens failed");
        events.push_back(e);
    }
    check(!events.empty(),"schedule empty");return events;
}
static void fill_schedule(llama_batch & batch,const ScheduleEvent & e) {
    batch.n_tokens=e.width;
    for(int col=0;col<e.width;++col) {
        batch.token[col]=e.tokens[col];batch.pos[col]=e.pos+col;batch.n_seq_id[col]=1;batch.seq_id[col][0]=0;batch.logits[col]=true;
    }
}
static void warm_schedule(EvalState & state,llama_batch & batch,const std::string & path) {
    for(const auto & e:read_schedule(path)) {
        check(llama_memory_seq_rm(llama_get_memory(state.ctx),0,e.pos,-1),"warm rollback failed");
        fill_schedule(batch,e);decode(state,batch,true);
    }
}
