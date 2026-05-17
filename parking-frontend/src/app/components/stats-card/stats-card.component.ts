import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-stats-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './stats-card.component.html',
  styleUrls: ['./stats-card.component.scss'],
})
export class StatsCardComponent {
  @Input() title: string = '';
  @Input() value: string | number = 0;
  @Input() subtitle: string = '';
  @Input() icon: string = '📊';
  @Input() color: 'blue' | 'green' | 'red' | 'orange' = 'blue';
}
