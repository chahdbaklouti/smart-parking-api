package com.example.smartparkingbackend.service;

import com.example.smartparkingbackend.model.OccupancyHistory;
import com.example.smartparkingbackend.repository.OccupancyRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OccupancyService {

    private final OccupancyRepository occupancyRepository;

    public OccupancyService(OccupancyRepository occupancyRepository) {
        this.occupancyRepository = occupancyRepository;
    }

    public OccupancyHistory getLatestOccupancy(String parkingId) {
        return occupancyRepository.findTop1ByParkingIdOrderByTimestampDesc(parkingId)
                .stream()
                .findFirst()
                .orElse(null);

    }

    public List<OccupancyHistory> getHistory(String parkingId) {
        return  occupancyRepository.findByParkingIdOrderByTimestampDesc(parkingId);
    }

}
