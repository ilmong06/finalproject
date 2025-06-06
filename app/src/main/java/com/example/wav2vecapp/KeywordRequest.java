package com.example.wav2vecapp;

public class KeywordRequest {

    private String uuid;

    public KeywordRequest(String uuid) {
        this.uuid = uuid;  // 외부에서 받은 uuid를 필드에 저장
    }

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }
}
