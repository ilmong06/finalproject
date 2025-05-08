package com.example.wav2vecapp;

public class GpsRequest {
    private String uuid;
    private double latitude;
    private double longitude;

    public GpsRequest(String uuid, double latitude, double longitude) {
        this.uuid = uuid;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    // Getter & Setter (필수는 아님, 필요 시 추가 가능)
    public String getUuid() {
        return uuid;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }
}
