"""Tests for phrase detection, extractors, gestures, and mood math."""

from maupo.triggers import (analyze_emotional_shift, carry_momentum,
                            detect_explicit_search_trigger,
                            detect_face_request, detect_face_reset,
                            detect_hard_memory_trigger, detect_self_edit_request,
                            detect_soft_memory_trigger, detect_state_question,
                            extract_hard_memory_fact,
                            extract_search_query, pick_face_gesture)


class TestHardMemoryTriggers:
    def test_imperative_teaching_phrases_trigger(self):
        assert detect_hard_memory_trigger("never forget that i hate olives")
        assert detect_hard_memory_trigger("always remember my birthday is march 3")
        assert detect_hard_memory_trigger("listen to me, this matters")
        assert detect_hard_memory_trigger("remember that i run marathons")

    def test_reminiscence_is_not_teaching(self):
        # "i remember that day" is nostalgia, not an instruction (invariant 4).
        assert not detect_hard_memory_trigger("i remember that day we met")

    def test_extraction_strips_trigger_phrase(self):
        fact = extract_hard_memory_fact("never forget that i hate olives")
        assert fact == "that i hate olives"


class TestSoftMemoryTriggers:
    def test_recall_phrases_trigger(self):
        assert detect_soft_memory_trigger("do you remember my sister's job")
        assert detect_soft_memory_trigger("remember when we talked about rust")
        assert detect_soft_memory_trigger("what did we say about porto")

    def test_plain_statements_do_not_trigger(self):
        assert not detect_soft_memory_trigger("my sister moved to porto")


class TestSearchTriggers:
    def test_explicit_search_phrases(self):
        assert detect_explicit_search_trigger("look up the gta 6 release date")
        assert detect_explicit_search_trigger("search for rust vs go")
        assert detect_explicit_search_trigger("google that")

    def test_query_extraction(self):
        assert extract_search_query("look up the gta 6 release date") == "the gta 6 release date"
        assert extract_search_query("search for rust vs go") == "rust vs go"


class TestFaceRequests:
    def test_direct_requests_detected(self):
        assert detect_face_request("smile for me") == "smile"
        assert detect_face_request("can you wink") == "wink"
        assert detect_face_request("SMILE") == "smile"
        assert detect_face_request("go on, smirk then") == "smirk"

    def test_negations_suppress(self):
        assert detect_face_request("don't smile") is None
        assert detect_face_request("stop laughing") is None
        assert detect_face_request("never cry again") is None

    def test_no_request_no_gesture(self):
        assert detect_face_request("what's the tallest mountain") is None
        assert detect_face_request("tell me a story") is None

    def test_inflections_map_to_the_sprite(self):
        # "stop smiling" has to be understood as being about the smile, even
        # though the negation then suppresses the gesture itself.
        assert detect_face_request("are you smiling at me") == "smile"
        assert detect_face_request("you're laughing") == "laugh"
        assert detect_face_request("she pouted") == "pout"
        assert detect_face_request("stop smiling") is None
        assert detect_face_request("what's the tallest mountain") is None


class TestSelfEditRequests:
    """A request to change HIM goes to the self-edit gate, not to the model."""

    def test_self_edits_are_detected(self):
        assert detect_self_edit_request("change yourself to be more sarcastic")
        assert detect_self_edit_request("change how you talk to me")
        assert detect_self_edit_request("rewrite your personality")
        assert detect_self_edit_request("edit your speech")
        assert detect_self_edit_request("i want you to be less formal, adjust your tone")
        assert detect_self_edit_request("make yourself less chatty")
        assert detect_self_edit_request("can you improve your greetings")
        assert detect_self_edit_request("self-edit your voice")

    def test_natural_change_requests_reach_the_gate(self):
        """Everyday ways to ask for a change must not fall through to chat."""
        assert detect_self_edit_request("be more sarcastic with me from now on")
        assert detect_self_edit_request("be a bit friendlier when i come home")
        assert detect_self_edit_request("sound a little warmer")
        assert detect_self_edit_request("i want you to be a little more playful")
        assert detect_self_edit_request("become more curious, ask me things more often")
        assert detect_self_edit_request("change the way you talk to me")
        assert detect_self_edit_request("change something about yourself")
        assert detect_self_edit_request("make your greetings warmer when i come back")
        assert detect_self_edit_request("make your voice softer")
        assert detect_self_edit_request("you should be more direct with me")
        assert detect_self_edit_request("talk less formally please")

    def test_ordinary_messages_are_not_self_edits(self):
        assert not detect_self_edit_request("i changed my mind about dinner")
        assert not detect_self_edit_request("you changed my life")
        assert not detect_self_edit_request("change the subject")
        assert not detect_self_edit_request("how's your day going")
        assert not detect_self_edit_request("edit this email for me")
        assert not detect_self_edit_request("i want to grow as a person")

    def test_talk_about_the_answer_is_not_a_body_edit(self):
        """Asking for a better ANSWER must stay ordinary conversation."""
        assert not detect_self_edit_request("can you be more specific about that?")
        assert not detect_self_edit_request("please be more clear next time")
        assert not detect_self_edit_request("you should be more careful with numbers")
        assert not detect_self_edit_request("that's more like it")
        assert not detect_self_edit_request("talk to me more about your day")
        assert not detect_self_edit_request("you talk too much")
        assert not detect_self_edit_request("i should be more careful")


class TestStateQuestions:
    """Direct questions about HIS state earn the honesty nudge; lookalike
    sentences that merely contain the same words must not."""

    def test_direct_state_questions_detected(self):
        assert detect_state_question("how are you")
        assert detect_state_question("how r u")
        assert detect_state_question("how are you feeling right now")
        assert detect_state_question("what are you feeling right now")
        assert detect_state_question("why arent you happy")
        assert detect_state_question("why are you so quiet")
        assert detect_state_question("are you ok")
        assert detect_state_question("are you sad")
        assert detect_state_question("you seem rather quiet")
        assert detect_state_question("dont be in a neutral mood")
        assert detect_state_question("why so quiet")
        assert detect_state_question("what's it like being you")

    def test_questions_that_merely_contain_the_words_are_not_state_questions(self):
        assert not detect_state_question("how are you so good at this game")
        assert not detect_state_question("are you okay with pizza tonight")
        assert not detect_state_question("how are you doing with the move")
        assert not detect_state_question("i don't be late, don't worry")
        assert not detect_state_question("don't be sad that it's over")
        assert not detect_state_question("what do you feel like eating")

    def test_ordinary_topics_are_not_state_questions(self):
        assert not detect_state_question("tell me about red dead redemption")
        assert not detect_state_question("why is the sky blue")
        assert not detect_state_question("smile for a moment")
        assert not detect_state_question("U SURE ABOUT THAT?")


class TestFaceReset:
    """A held expression must be releasable by the person who asked for it."""

    def test_return_to_normal_face_is_a_reset(self):
        assert detect_face_reset("return to normal face expression")
        assert detect_face_reset("back to your normal face")
        assert detect_face_reset("neutral face please")
        assert detect_face_reset("drop the smirk")
        assert detect_face_reset("wipe that grin off")
        assert detect_face_reset("stop smiling")

    def test_ordinary_sentences_are_not_resets(self):
        assert not detect_face_reset("the normal distribution is symmetric")
        assert not detect_face_reset("i had a long day at work")
        assert not detect_face_reset("smile for me")   # a request, not a reset
        assert not detect_face_reset("don't smile")


class TestPickFaceGesture:
    def test_laughter_slang_wins(self):
        assert pick_face_gesture(0.0, 0.0, 0.2, 0.2, "lmaooo that's wild") == "laugh"

    def test_grief_words_cry(self):
        assert pick_face_gesture(0.0, 0.0, -0.5, -0.5, "i went to a funeral today") == "cry"

    def test_big_positive_swing_smiles_or_laughs(self):
        assert pick_face_gesture(0.2, 0.0, 0.5, 0.6, "amazing news") == "laugh"
        assert pick_face_gesture(0.2, 0.0, 0.5, 0.1, "that's great") == "smile"

    def test_negative_swing_pouts_or_cries(self):
        assert pick_face_gesture(-0.2, 0.0, -0.5, -0.4, "awful day") == "cry"
        assert pick_face_gesture(-0.2, 0.0, -0.3, 0.2, "that's annoying") == "pout"

    def test_sarcasm_smirks(self):
        assert pick_face_gesture(0.0, 0.0, 0.1, 0.1, "yeah right, sure sure") == "smirk"

    def test_shock_surprises(self):
        assert pick_face_gesture(0.0, 0.2, 0.2, 0.5, "omg no way!!") == "surprised"

    def test_quiet_message_gives_nothing(self):
        assert pick_face_gesture(0.0, 0.0, 0.0, 0.0, "ok") is None


class TestAnalyzeEmotionalShift:
    def test_positive_message_lifts_valence(self):
        v_delta, e_delta = analyze_emotional_shift("i'm so happy and excited today", 0.0, 0.0)
        assert v_delta > 0
        assert e_delta > 0

    def test_negative_message_drops_valence(self):
        v_delta, _ = analyze_emotional_shift("everything is terrible and i'm exhausted", 0.0, 0.0)
        assert v_delta < 0

    def test_momentum_pulls_back_to_neutral(self):
        v_delta, e_delta = analyze_emotional_shift("just chatting", 0.8, -0.6)
        assert v_delta < 0  # momentum pushes back toward zero
        assert e_delta > 0

    def test_neutral_message_only_momentum(self):
        v_delta, e_delta = analyze_emotional_shift("the sky is blue", 0.0, 0.0)
        assert v_delta == 0.0
        assert e_delta == 0.0


class TestCarryMomentum:
    """Sustained emotion compounds (half-life one turn); one spike never does."""

    def test_repeated_moderate_pushes_compound(self):
        m = (0.0, 0.0)
        for _ in range(4):
            m, (cv, _ce) = carry_momentum(m, (0.2, 0.0))
        assert cv > 0.05  # sustained moderate positivity keeps growing

    def test_single_spike_is_never_amplified(self):
        m, (cv, ce) = carry_momentum((0.0, 0.0), (0.5, 0.5))
        assert cv == 0.0 and ce == 0.0  # a big one-off push carries nothing

    def test_negative_sustained_carries_negative(self):
        m = (0.0, 0.0)
        for _ in range(4):
            m, (cv, _ce) = carry_momentum(m, (-0.2, 0.0))
        assert cv < -0.05  # ongoing hurt builds too, not just hype

    def test_carry_dies_out_when_pushes_stop(self):
        m = (0.3, 0.0)
        m, (cv, _ce) = carry_momentum(m, (0.0, 0.0))  # one quiet turn
        m, (cv, _ce) = carry_momentum(m, (0.0, 0.0))  # another
        assert cv == 0.0  # momentum decayed below the carry floor
