# Write your first design

In this tutorial you build a jurat design for a shop's help desk, decide two customer
messages offline, and replay the records. You need Python 3.11 or later and jurat 0.3.0.

## The design you'll build, and where it lives

The help desk asks what a message is about, whether it is urgent and which order number it
quotes, then routes or escalates it. The finished files are in
`docs/tutorials/write_a_design/`: `helpdesk_design.py` holds the design and
`helpdesk_run.py` runs it. If you installed jurat from a package index, fetch the two
files from the repository at
[`docs/tutorials/write_a_design/`](https://github.com/CRSiegfried/jurat/tree/main/docs/tutorials/write_a_design).

Keep the design in its own module and import it from your script, as this example does. A
component's module is part of the design's hash, so the same design run as the launched
script, `__main__`, gets a different hash and its records replay with `design-mismatch`
(the spec's [Replay](../spec/README.md#replay) section says how to find the cause).

## Sources and oracles

A source fetches answers from outside. It declares the refs it provides, the board values
it needs, and the oracle that answers it.

1. Open `helpdesk_design.py` and find the first source, `triage`, which asks two questions:

   ```python
   SOURCES = {
       "triage": Source(
           {
               "q:triage/urgent": Question("noul", Text("Does the customer need an answer today?")),
               "q:triage/topic": Question(
                   "choice", Text("What is the customer's message about?"), TOPICS
               ),
           },
           needs=["t:state"],
           oracle="classifier",
       ),
   ```

   The key, `triage`, is the source's id. Sources and rules share one set of names.

2. `needs=["t:state"]` hands the source the turn's state. `oracle="classifier"` names the
   binding that answers it: at decide you give one binding per oracle name. A source with no
   `oracle` is answered by the binding named `default`.

3. Find the second source, `lookup`, with the same `needs` and `oracle="reader"`. Its one
   question, `q:order/number`, comes in the next section.

4. At the bottom, the design's `state=t.record(message=t.string)` makes each turn's state
   a record of one string, `message`.

## Questions

Each `q:` ref maps to a `Question`: its kind, its wording as a `Text`, and any labels.

1. The urgency question is a `noul`, a yes-or-no question. It takes no labels.

2. The topic question is a `choice`. Its labels can be plain names, such as
   `["delivery", "return"]`, but a name alone leaves the model to guess what it covers. A
   described label is exactly `{"name": ..., "description": ...}`:

   ```python
   TOPICS = [
       {"name": "delivery", "description": "Where an order is, or when it will arrive"},
       {"name": "return", "description": "Sending something back, or a refund for it"},
       ...,  # "damaged" and "other", described the same way
   ]
   ```

   Answers and rules use the names alone. The description travels with the question to the
   binding, and changing it makes a different question.

3. Find `lookup`'s extraction question, which asks for a typed value read from a text:

   ```python
               "q:order/number": Question(
                   "extraction",
                   Text("The order number the customer quotes"),
                   None,
                   value_type="text",
                   text={"ref": "t:state.message"},
               ),
   ```

   Its labels are `None`; [the spec](../spec/README.md#the-extraction-kind) lists the
   value types. `text` is an origin, `{"ref": r}`, naming the value the model reads.

4. The `text` ref must name a value whose type is exactly `string`. `t:state.message` is
   one, so the design loads; declare `message` as `t.integer()` and `load` refuses it.

5. The `text` ref joins the source's needs: `lookup` declared only `t:state`, but its
   binding also receives `t:state.message`.

## Rules

A rule turns answers into an estimate, a `d:` ref the policy reads. It is a function of a
read view, `r`.

1. Find the topic rule:

   ```python
   RULES = {
       "topic": Rule(
           "d:topic",
           t.label("damaged", "delivery", "other", "return"),
           lambda r: pick_with_margin(r, "q:triage/topic", 0.5, 0.2),
           version="1",
           reads=["q:triage/topic"],
       ),
   ```

   The rule provides `d:topic`, one of the four names, listed in code-point order.

2. Every rule has a `version` and its `reads`. `load` refuses a rule with no version
   string; raise it when the function changes. `reads` lists every ref the function reads;
   an undeclared read makes the rule's ref `Unavailable` with the cause `undeclared-read`.

3. `pick_with_margin(r, ref, bar, margin)` returns the top label when its probability is at
   least `bar` and leads the second by at least `margin`. Otherwise it abstains, returning
   `Unsure` with the cause `abstained` naming `ref`. A marker it reads, it returns as is.

4. The urgency rule is built the same way, with `t.label("no", "yes")` and a bar of 0.6.
   Its key, `urgent`, is the name of the `d:` ref it provides, as every rule's must be.

5. The order rule, `d:order` of type `t.string`, calls
   `extract(r, "q:order/number", 0.5)`. `extract(r, ref, null_bar)` returns the answer's
   `value`. It abstains, again with `Unsure` and `abstained`, when the value is `None` or
   the model's `p_null`, the probability that the text holds none, is at or above
   `null_bar`.

An estimate jurat cannot settle is never `None` or a quiet `no` but a marker: `Unsure`,
`Unavailable` or `Unasked`. Test for them with `isinstance`:

```python
MARKERS = (Unsure, Unavailable, Unasked)
```

## Policy and plans

The policy is one function that reads the estimates and returns the plan.

1. Find it in `helpdesk_design.py`:

   ```python
   def route(r: ReadView) -> Plan:
       """Escalate what the desk cannot settle or must answer today; inform the rest."""
       topic = r.read("d:topic")
       urgent = r.read("d:urgent")
       order = r.read("d:order")
       if isinstance(topic, MARKERS):
           return Plan("escalate", reason="the topic is not settled")
       if urgent == "yes":
           return Plan("escalate", reason=f"an urgent {topic} message")
       if isinstance(order, MARKERS):
           return Plan("inform", reply=f"Routed to {topic}; no order number found.")
       return Plan("inform", reply=f"Routed to {topic} for order {order}.")
   ```

   Each marker is handled on purpose: an unsettled topic escalates, an unsettled urgency
   never equals `"yes"`, and a missing order number still informs.

2. Each plan carries the text its kind needs. An `inform` plan needs a `reply`, the text
   the caller sends back. An `escalate` plan needs a `reason`. A plan without its text, or
   with a `reply` on an `escalate` plan, becomes a `broken` plan whose cause is
   `invalid-return`. Dispatch on `plan.kind`, never on the verdict alone: a `broken` plan
   has no steps, and the guard passes it with nothing to run.

3. The design wraps it as
   `Policy(route, version="1", reads=["d:topic", "d:urgent", "d:order"])` and passes it,
   with `state`, `sources` and `rules`, to `Design`.

## Bindings and decide

A binding is the async function that answers a batch of sources. This one returns fixed
answers, so no model is needed.

1. Open `helpdesk_run.py` and read `fixed_answers`. It takes a tuple of `SourceCall`s and
   returns a `CallResult` whose `values` map each `q:` ref to its answer. A `noul` or
   `choice` answer gives a probability to each label name, such as
   `{"yes": 0.2, "no": 0.8}`. An extraction answer has exactly five keys:

   ```python
   start = call.needs[question.text].find("A-1042")
   found = start >= 0
   values[ref] = {
       "value": "A-1042" if found else None,
       "guess": None,
       "p_null": 0.02 if found else 0.95,
       "confidence": 0.9,
       "span": [start, start + 6] if found else None,
   }
   ```

   `question.text` is the key in `call.needs` that holds the message, and `span` is a pair
   of offsets into it.

2. A binding receives a source's questions in the code-point order of their refs, so
   `q:triage/topic` comes first, since that order is part of what ran.

3. Read `main`. It loads the design, binds both oracle names, and decides each message:

   ```python
   loaded = load(DESIGN)
   bindings = {"classifier": fixed_answers, "reader": fixed_answers}
   decision = decide_sync(loaded, {"message": message}, bindings)
   ```

   `load` raises `InvalidDesign` naming every failure. Without `"reader"`, `decide_sync`
   raises `EntryRefusal`. A real binding, such as one over a model provider's API, plugs
   in at the same place, in this mapping. So does `jurat.from_record(record, loaded)`,
   which answers an oracle's sources from a stored record while the others run live;
   `record` is a `Record` or a stored document, bytes or parsed, read through `read_record`.

4. Run the driver from its directory:

   ```
   cd docs/tutorials/write_a_design
   python helpdesk_run.py
   ```

   Each turn's first line shows the plan's kind, its text and the guard's verdict:

   ```
   inform | Routed to delivery for order A-1042. | passed
   inform | Routed to delivery; no order number found. | passed
   ```

   In the second, `extract` abstained, so `d:order` is `Unsure`.

## Reading the record

`decision.record.document` is the record as canonical JSON bytes; store them and read them
back with `read_record`. Its `log` is the hashed turn, one item per step; `unhashed` sits
outside every hash.

1. Find the call items in the log. A call item holds `item`, `round`, `sources`,
   `questions`, `outcome`, `snapshot`, `elicitation`, `prompt` and `precision`, all hashed.

2. `start`, `latency`, `usage`, `attempts`, `warnings`, `truncated`, `sums`, `unexpected`
   and `invalid` sit under `unhashed["calls"]`, keyed by the call item's log index as a
   string.

3. Name each call's oracle. A call item names no oracle or provider, so look up each of its
   `sources` in the design: the source's `oracle`, or `default` when it declares none. The
   provider shows in `prompt` or `snapshot`, null here. The driver does it:

   ```python
   unhashed = json.loads(stored)["unhashed"]
   for index, item in enumerate(record.log):
       if item["item"] == "call":
           oracles = [DESIGN.sources[s].oracle or "default" for s in item["sources"]]
           latency = unhashed["calls"][str(index)]["latency"]
           print(f"  call {index}: {item['sources']} via {oracles}, {latency:.6f} s")
   ```

   It prints `call 3: ['lookup'] via ['reader']` and `call 4: ['triage'] via
   ['classifier']`, each with its latency.

## Replay

1. Replay each record, as `main` does: `rerun, findings = replay_sync(record, loaded)`.
   Replay answers from the record and calls nothing. Each finding is one difference.

2. Check the findings, and the plan and verdict, not `record_hash`, since replay answers
   with null provenance and a real provider's record hashes differently from its replay.
   This offline binding returns null provenance too, so here the hashes happen to match:
   the driver prints `same hash: True`.

3. Each turn prints `replay findings: []`: the same answers rebuilt the same plan.

To compare two models' records of one turn, `jurat.evaluation.use_answers(record, loaded,
other)` reruns `record` with `other`'s answers. It is async, and `use_answers_sync` is its
twin. Each record is a `Record` or a stored document. It returns
`{"other": <other's record hash>, "findings": [...]}`, the findings being replay's.
