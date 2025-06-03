package com.example.wav2vecapp;

public class ReportItem {
    private String date;
    private String location;
    private String keyword;

    public ReportItem(String date, String location, String keyword) {
        this.date = date;
        this.location = location;
        this.keyword = keyword;
    }

    public String getDate() {
        return date;
    }

    public String getLocation() {
        return location;
    }

    public String getKeyword() {
        return keyword;
    }
}

