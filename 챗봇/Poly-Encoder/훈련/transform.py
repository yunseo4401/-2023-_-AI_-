class SelectionSequentialTransform(object):
    def __init__(self, tokenizer, max_len):
        # 클래스 초기화 함수
        # tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저
        # max_len: 반환될 토큰 시퀀스의 최대 길이
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, texts):
        # 호출 가능한 메서드, 입력 텍스트 리스트를 받아 토큰화 및 패딩된 시퀀스를 생성하고 반환
        input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list = [], [], [], []
        for text in texts:
            # 각 텍스트에 대해 토큰화하고 패딩된 딕셔너리를 생성
            tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True)
            input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
            # 생성된 시퀀스가 최대 길이와 일치하는지 확인
            assert len(input_ids) == self.max_len
            assert len(input_masks) == self.max_len
            # 생성된 시퀀스를 리스트에 추가
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)

        # 생성된 시퀀스 리스트를 반환
        return input_ids_list, input_masks_list


class SelectionJoinTransform(object):
    def __init__(self, tokenizer, max_len):
        # 클래스 초기화 함수
        # tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저
        # max_len: 반환될 토큰 시퀀스의 최대 길이
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 특수 토큰들의 ID를 가져오기
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=True)  # '\n' 토큰 추가
        self.pad_id = 0

    def __call__(self, texts):
        # 호출 가능한 메서드, 입력 텍스트 리스트를 하나의 텍스트로 결합하여 토큰화 및 패딩된 시퀀스를 생성하고 반환
        context = '\n'.join(texts)  # 입력 텍스트를 '\n'으로 연결하여 하나의 텍스트로 만듦
        tokenized_dict = self.tokenizer(context, padding='max_length', truncation=True, max_length=self.max_len)
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        # 생성된 시퀀스가 최대 길이와 일치하는지 확인
        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len

        # 생성된 시퀀스를 반환
        return input_ids, input_masks


class SelectionConcatTransform(object):
    def __init__(self, tokenizer, max_len):
        # 클래스 초기화 함수
        # tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저
        # max_len: 반환될 토큰 시퀀스의 최대 길이
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 특수 토큰들의 ID를 가져오기
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=True)  # '\n' 토큰 추가
        self.pad_id = 0

    def __call__(self, context, responses):
        # 호출 가능한 메서드, 컨텍스트와 응답 리스트를 받아 토큰화 및 패딩된 시퀀스를 생성하고 반환
        context = '\n'.join(context)  # 컨텍스트 텍스트를 '\n'으로 연결하여 하나의 텍스트로 만듦
        tokenized_dict = self.tokenizer.encode_plus(context, padding='max_length', truncation=True)
        context_ids, context_masks, context_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
        ret_input_ids = []
        ret_input_masks = []
        ret_segment_ids = []
        for response in responses:
            tokenized_dict = self.tokenizer.encode_plus(response)
            response_ids, response_masks, response_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
            response_segment_ids = [1]*(len(response_segment_ids)-1)
            input_ids = context_ids + response_ids[1:]
            input_ids = input_ids[-self.max_len:]
            input_masks = context_masks + response_masks[1:]
            input_masks = input_masks[-self.max_len:]
            input_segment_ids = context_segment_ids + response_segment_ids
            input_segment_ids = input_segment_ids[-self.max_len:]
            input_ids[0] = self.cls_id
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))
            input_masks += [0] * (self.max_len - len(input_masks))
            input_segment_ids += [0] * (self.max_len - len(input_segment_ids))
            # 생성된 시퀀스가 최대 길이와 일치하는지 확인
            assert len(input_ids) == self.max_len
            assert len(input_masks) == self.max_len
            assert len(input_segment_ids) == self.max_len
            # 생성된 시퀀스를 리스트에 추가
            ret_input_ids.append(input_ids)
            ret_input_masks.append(input_masks)
            ret_segment_ids.append(input_segment_ids)