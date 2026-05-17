import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, switchMap, startWith } from 'rxjs';
import {
  Camera,
  CameraListResponse,
  GlobalStats,
  ParkingStats,
  SpringGlobalStats,
  TimeSlotStats,
  ParkingInfo,
} from '../models/parking.model';

// Service Angular pour interagir avec les APIs Flask (temps réel) et Spring Boot (historique)- Angular NE parle jamais directement au backend - Il passe toujours par ce service
@Injectable({
  providedIn: 'root',
})
export class ParkingService {
  // URLs des APIs
  private flaskUrl = 'http://localhost:5000'; // Flask - streaming & stats live
  private springUrl = 'http://localhost:8080'; // Spring Boot - historique

  constructor(private http: HttpClient) {}

  // ═══════════════════════════════════════════════════════════════
  //                    FLASK APIs (Temps réel)
  // ═══════════════════════════════════════════════════════════════

  /**
   * Récupère la liste des caméras disponibles
   */
  getCameras(): Observable<CameraListResponse> {
    const url = `${this.flaskUrl}/api/cameras`;
    console.log('📡 ParkingService.getCameras() - URL:', url);
    return this.http.get<CameraListResponse>(url);
  }

  /**
   * Récupère les stats en temps réel de tous les parkings
   */
  getLiveStats(): Observable<GlobalStats> {
    return this.http.get<GlobalStats>(`${this.flaskUrl}/api/stats`);
  }

  /**
   * Récupère les stats en temps réel d'un parking spécifique
   */
  getLiveParkingStats(parkingId: string): Observable<ParkingStats> {
    return this.http.get<ParkingStats>(`${this.flaskUrl}/api/stats/${parkingId}`);
  }

  /**
   * Retourne l'URL du stream vidéo pour une caméra
   * À utiliser directement dans <img [src]="...">
   */
  getVideoStreamUrl(parkingId: string): string {
    return `${this.flaskUrl}/video/${parkingId}`;
  }

  /**
   * Polling des stats toutes les N secondes
   */
  pollLiveStats(intervalMs: number = 3000): Observable<GlobalStats> {
    return interval(intervalMs).pipe(
      startWith(0),
      switchMap(() => this.getLiveStats()),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //                 SPRING BOOT APIs (Historique)
  // ═══════════════════════════════════════════════════════════════

  /**
   * Liste tous les parkings avec stats actuelles (depuis DB)
   */
  getAllParkings(): Observable<ParkingInfo[]> {
    return this.http.get<ParkingInfo[]>(`${this.springUrl}/api/parking`);
  }

  /**
   * Stats globales depuis la base de données
   */
  getGlobalStats(): Observable<SpringGlobalStats> {
    return this.http.get<SpringGlobalStats>(`${this.springUrl}/api/parking/stats`);
  }

  /**
   * Stats moyennes par heure pour un parking
   */
  getHourlyStats(parkingId: string): Observable<TimeSlotStats[]> {
    return this.http.get<TimeSlotStats[]>(
      `${this.springUrl}/api/parking/${parkingId}/stats/hourly`,
    );
  }

  /**
   * Stats moyennes par jour pour un parking
   */
  getDailyStats(parkingId: string): Observable<TimeSlotStats[]> {
    return this.http.get<TimeSlotStats[]>(`${this.springUrl}/api/parking/${parkingId}/stats/daily`);
  }
}
