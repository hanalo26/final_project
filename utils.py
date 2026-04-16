def print_tool_calls(result, detail=False):
    """Agent의 도구 호출 내역을 출력하는 공통 함수입니다.

    Args:
        result: agent.run()의 반환값입니다.
        detail: True이면 메시지 타입과 내용 미리보기까지 출력합니다.
                False이면 호출된 도구 이름만 간략히 출력합니다.
    """
    messages = result.all_messages()

    if detail:
        # 상세 모드: 메시지 타입, 도구 호출 파라미터, 도구 응답까지 모두 출력합니다
        print(f"총 {len(messages)}개의 메시지가 교환되었습니다.")
        print()
        for msg in messages:
            msg_type = type(msg).__name__
            print(f"[{msg_type}]")
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    part_type = type(part).__name__
                    if part_type == 'ToolCallPart':
                        # AI가 도구를 호출한 부분: 도구 이름 + 전달한 파라미터
                        args_str = str(part.args)
                        if len(args_str) > 200:
                            args_str = args_str[:200] + '...'
                        print(f"  [도구 호출] {part.tool_name}({args_str})")
                    elif part_type == 'ToolReturnPart':
                        # 도구가 반환한 결과
                        content_str = str(part.content)
                        if len(content_str) > 300:
                            content_str = content_str[:300] + '...'
                        print(f"  [도구 응답] {part.tool_name} => {content_str}")
                    elif hasattr(part, 'content'):
                        # 일반 텍스트 (사용자 질문, AI 최종 답변)
                        content_preview = str(part.content)[:150]
                        print(f"  [텍스트] {content_preview}...")
            print()
    else:
        # 간략 모드: 호출된 도구 이름과 응답 내용만 출력합니다
        print("[호출된 도구 목록]")
        for msg in messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'tool_name'):
                        print(f"  - {part.tool_name}")
                    elif hasattr(part, 'content'):
                        content_str = str(part.content)
                        if len(content_str) > 200:
                            content_str = content_str[:200] + '...'
                        print(f"  내용: {content_str}")
