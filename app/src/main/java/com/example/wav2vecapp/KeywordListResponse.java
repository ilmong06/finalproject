package com.example.wav2vecapp;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class KeywordListResponse {

    @SerializedName("keywd_text")
    private String keywd_text;

    @SerializedName("add_date")
    private String add_date;

    public String getKeyword(){
        return keywd_text;
    }

    public String date(){
        return add_date;
    }
}
